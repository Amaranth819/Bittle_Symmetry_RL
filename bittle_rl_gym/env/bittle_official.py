from isaacgym import gymapi, gymtorch
from bittle_rl_gym.env.base_task import BaseTask
from typing import List
from isaacgym.torch_utils import *
from bittle_rl_gym.utils.helpers import class_to_dict, get_euler_xyz_in_tensor
from bittle_rl_gym.cfg.bittle_official_config import BittleOfficialConfig
import os
import torch
import pickle
import matplotlib.pyplot as plt
import scipy
from scipy.stats import vonmises_line
from collections import defaultdict, deque
from gym.spaces import Box


CAMERA_WIDTH = 512
CAMERA_HEIGHT = 384 
MAX_VIDEO_LENGTH = 1000
VIDEO_FPS = 60
SAVE_HISTORY_LENGTH = 1


class BittleOfficial(BaseTask):
    def __init__(self, cfg : BittleOfficialConfig, sim_params, physics_engine, sim_device, headless : bool, record_video : bool = False):
        self.cfg = cfg
        self.sim_params = sim_params
        self._parse_cfg()

        # To setup rl_games
        self.observation_space = Box(shape = (self.cfg.env.num_observations,), low = -np.inf, high = np.inf)
        self.state_space = Box(shape = (self.cfg.env.num_observations,), low = -np.inf, high = np.inf)
        self.action_space = Box(shape = (self.cfg.env.num_actions,), low = -self.cfg.normalization.clip_actions, high = self.cfg.normalization.clip_actions)
        self.num_states = self.state_space.shape[-1]

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, record_video)

        self._init_buffers()
        self._init_foot_periodicity_buffer()
        self._init_foot_periodicity_visualization()
        self.reward_functions = self._prepare_reward_functions(class_to_dict(self.cfg.rewards))
        self.episode_rew_sums = {name : torch.zeros_like(self.rew_buf) for name in list(self.reward_functions.keys()) + ['total']}

        # Set viewer camera
        if not self.headless:
            self._set_viewer_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat, self.cfg.viewer.ref_env)

        # Cameras (currently for video recording)
        self.cameras = {} 
        self.camera_tensors = []
        self.camera_frames = defaultdict(lambda: deque(maxlen = MAX_VIDEO_LENGTH))


    def __del__(self):
        self.save_record_video(name = 'video', postfix = 'mp4')
        npy_file_name = 'fp'
        self._save_foot_periodicity_visualization(file_name = npy_file_name)
        plot_foot_periodicity(f'{npy_file_name}.pkl', fig_name = 'fp')


    '''
        Get some parameters from the configuration file.
    '''
    def _parse_cfg(self):
        self.dt = self.sim_params.dt
        self.max_episode_length_in_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_in_s / self.dt)
        print('Max episode length:', self.max_episode_length)
        self.auto_PD_gains = self.cfg.control.auto_PD_gains
        self.reward_cfg = self.cfg.rewards
        self.command_cfg = self.cfg.commands
        self.domain_rand_cfg = self.cfg.domain_rand
        self.foot_periodicity_cfg = self.cfg.foot_periodicity


    '''
        Functions for setting cameras and video recording.
    '''
    def _set_viewer_camera(self, pos : List[int], lookat : List[int], ref_env_idx : int = -1):
        """
            Set the camera position and direction.
            Input:
                pos: (x, y, z)
                lookat: (x, y, z)
        """
        assert len(pos) == 3 and len(lookat) == 3
        cam_pos, cam_target = gymapi.Vec3(*pos), gymapi.Vec3(*lookat)
        if ref_env_idx >= 0:
            env_handle = self.envs[ref_env_idx]
            # Set the camera to track a certain environment.
            ref_env_base_pos = gymapi.Vec3(*self.root_states[ref_env_idx, :3])
            cam_pos = cam_pos + ref_env_base_pos
            cam_target = cam_target + ref_env_base_pos
        else:
            env_handle = None
        self.gym.viewer_camera_look_at(self.viewer, env_handle, cam_pos, cam_target)


    def _create_camera(self, env_idx, p = [0, -0.3, 0], axis = [0, 0, 1], angle = 90.0, follow = 'FOLLOW_POSITION'):
        # If you want to record any videos, call _create_camera() to create a camera in the scene.
        camera_props = gymapi.CameraProperties()
        camera_props.width, camera_props.height = CAMERA_WIDTH, CAMERA_HEIGHT
        camera_props.enable_tensors = True
        camera_idx = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*p)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(*axis), np.radians(angle))
        self.gym.attach_camera_to_body(camera_idx, self.envs[env_idx], 0, local_transform, getattr(gymapi, follow))

        self.cameras[camera_idx] = env_idx
        camera_ts = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], camera_idx, gymapi.IMAGE_COLOR) # IMAGE_COLOR - 4x 8 bit unsigned int - RGBA color
        camera_ts = gymtorch.wrap_tensor(camera_ts)
        self.camera_tensors.append(camera_ts)


    def _render_cameras(self):
        if len(self.cameras) > 0:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            for idx in range(len(self.camera_tensors)):
                self.camera_frames[idx].append(self.camera_tensors[idx].cpu().numpy())
            self.gym.end_access_image_tensors(self.sim)


    def save_record_video(self, name, postfix = 'mp4'):
        if len(self.camera_frames) > 0:
            assert postfix in ['gif', 'mp4']
            import imageio
            import cv2
            
            for idx, video_frames in self.camera_frames.items():
                video_path = f'{name}_{idx}.{postfix}'
                if postfix == 'gif':
                    with imageio.get_writer(video_path, mode = 'I', duration = 1 / VIDEO_FPS) as writer:
                        for frame in video_frames:
                            writer.append_data(frame)
                elif postfix == 'mp4':
                    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT), True) # 
                    for frame in video_frames:
                        video.write(frame[..., :-1])
                    video.release()

                print(f'Save video to {video_path} ({len(video_frames)} frames, {len(video_frames) / VIDEO_FPS} seconds).')


    '''
        Initialize the buffers.
    '''
    def _init_buffers(self):
        # Actor root states: base positions and velocities
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # [num_envs, 13] containing position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        print('root_states:', self.root_states.size())

        # Degree of freedom states: positions and velocities
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        print('dof_pos:', self.dof_pos.size())
        self.dof_vel = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]
        print('dof_vel:', self.dof_vel.size())

        # Net contact forces: [num_rigid_bodies, 3]
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies, 3) # [num_envs, num_rigid_bodies, 3]
        print('contact_forces:', self.contact_forces.size())

        # Torques
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof) 
        print('torques:', self.torques.size())

        # Rigid body states
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13) # [num_envs, num_rigid_bodies, 13]
        print('rigid_body_states:', self.rigid_body_states.size())

        # PD gains
        self.P_gains = torch.zeros(self.num_dof, dtype = torch.float, device = self.device, requires_grad = False)
        for dof_key, kp in self.cfg.control.stiffness.items():
            dof_idx = self.dof_name2idx[dof_key]
            self.P_gains[dof_idx] = kp
        self.D_gains = torch.zeros(self.num_dof, dtype = torch.float, device = self.device, requires_grad = False)
        for dof_key, kd in self.cfg.control.damping.items():
            dof_idx = self.dof_name2idx[dof_key]
            self.D_gains[dof_idx] = kd

        # Actions
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype = torch.float, device = self.device, requires_grad = False)
        print('actions:', self.actions.size())

        # Noisy observations
        self.noise_obs_buf = torch.zeros_like(self.obs_buf)
        self.noise_obs_buf[..., 0:3] = self.cfg.domain_rand.observation.lin_vel_noise
        self.noise_obs_buf[..., 3:6] = self.cfg.domain_rand.observation.ang_vel_noise
        self.noise_obs_buf[..., 6:6 + self.num_dof] = self.cfg.domain_rand.observation.dof_pos_noise
        self.noise_obs_buf[..., 6 + self.num_dof:6 + 2*self.num_dof] = self.cfg.domain_rand.observation.dof_vel_noise
        self.noise_obs_buf[..., 6 + 2*self.num_dof:] = 0

        # Gravity direction
        self.gravity_vec = to_torch(get_axis_params(-1, self.up_axis_idx), device = self.device).repeat((self.num_envs, 1))

        # Command velocities
        self.command_lin_vel = torch.zeros((self.num_envs, 3), dtype = torch.float, device = self.device, requires_grad = False)
        self.command_ang_vel = torch.zeros_like(self.command_lin_vel)

        # Initial dof positions
        self.default_dof_pos = torch.zeros((self.num_dof), dtype = torch.float, device = self.device, requires_grad = False)
        for name, angle in self.cfg.init_state.default_joint_angles.items():
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[self.dof_name2idx[name]] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Save the properties at the last time step
        self.history_data = {}
        self.save_history_data_keys = ['root_states', 'dof_pos', 'dof_vel', 'torques', 'rigid_body_states', 'actions', 'P_gains', 'D_gains']
        for key in self.save_history_data_keys:
            self.history_data[key] = deque(maxlen = SAVE_HISTORY_LENGTH)

        # Information
        self.extras = {}

        # 2024.08.26: Store foot periodicity information to compute different rewards.
        self.E_C_frc = torch.zeros(size = (self.num_envs, len(self.foot_shank_indices)), dtype = torch.float32, device = self.device)
        self.E_C_spd = torch.zeros(size = (self.num_envs, len(self.foot_sole_indices)), dtype = torch.float32, device = self.device)

        # # Foot contacts and air time
        # self.feet_air_time = torch.zeros(self.num_envs, len(self.foot_sole_indices), dtype=torch.float, device=self.device, requires_grad=False)
        # self.last_contacts = torch.zeros(self.num_envs, len(self.foot_sole_indices), dtype=torch.bool, device=self.device, requires_grad=False)
    

    '''
        Get base information.
    '''
    def _get_base_pos(self, root_states):
        return root_states[..., 0:3]


    def _get_base_quat(self, root_states):
        return root_states[..., 3:7]
    

    def _get_base_rpy(self, root_states):
        quat = self._get_base_quat(root_states)
        return get_euler_xyz_in_tensor(quat)
    

    def _get_base_lin_vel(self, root_states):
        quat = self._get_base_quat(root_states)
        return quat_rotate_inverse(quat, root_states[..., 7:10])
    
    
    def _get_base_ang_vel(self, root_states):
        quat = self._get_base_quat(root_states)
        return quat_rotate_inverse(quat, root_states[..., 10:13])
    

    def _get_base_projected_gravity(self, root_states, gravity_vec):
        quat = self._get_base_quat(root_states)
        return quat_rotate_inverse(quat, gravity_vec)


    '''
        Update the PD gains for each environment.
    '''
    def _update_PD_gains(self, new_kp, new_kd):
        for i in range(self.num_envs):
            dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
            dof_props['stiffness'][:] = new_kp
            dof_props['damping'][:] = new_kd
            self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], dof_props)


    '''
        Domain randomization
    '''
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.domain_rand_cfg.rigid_shape_prop.apply:
            if env_id == 0:
                # prepare friction randomization
                friction_scale = self.domain_rand_cfg.rigid_shape_prop.friction_scale
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_scale[0], friction_scale[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction *= self.friction_coeffs[env_id]
        return props


    def _process_rigid_body_props(self, props, env_id):
        if self.domain_rand_cfg.rigid_body_prop:
            mass_scale = self.cfg.domain_rand.rigid_body_prop.mass_scale
            props[0].mass *= np.random.uniform(mass_scale[0], mass_scale[1])
        return props


    def _process_obs_noises(self):
        noise = torch.rand_like(self.noise_obs_buf) * 2 - 1.0
        return self.noise_obs_buf * noise


    """
        Create the simulation environment.
    """
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()


    def _create_ground_plane(self):
        """ 
            Add a ground plane to the simulation, set friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self):
        """
            Create #num_envs environment instances in total.
        """
        asset_cfg = self.cfg.asset

        # Load the urdf and mesh files of the robot.
        # The urdf is downloaded from https://github.com/PetoiCamp/ros_opencat/tree/ros1/petoi_ROS_model_docs.
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/bittle-official')
        asset_file = asset_cfg.file

        # Set some physical properties of the robot.
        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = asset_cfg.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = asset_cfg.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = asset_cfg.flip_visual_attachments
        asset_options.fix_base_link = asset_cfg.fix_base_link
        asset_options.density = asset_cfg.density
        asset_options.angular_damping = asset_cfg.angular_damping
        asset_options.linear_damping = asset_cfg.linear_damping
        asset_options.max_angular_velocity = asset_cfg.max_angular_velocity
        asset_options.max_linear_velocity = asset_cfg.max_linear_velocity
        asset_options.armature = asset_cfg.armature
        asset_options.thickness = asset_cfg.thickness
        asset_options.disable_gravity = asset_cfg.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        # Save body and dof names.
        self.body_idx2name = self.gym.get_asset_rigid_body_names(robot_asset)
        print('Body names:', self.body_idx2name) # ['base_link', 'servo_neck__1', 'c_thlf_1', 'servos_lf_1', 'shank_lf_1', 'c_thlr_1', 'servos_lr_1', 'shank_lr_1', 'c_thrf__1', 'servos_rf_1', 'shank_rf_1', 'c_thrr_1', 'servos_rr_1', 'shank_rr_1']
        self.body_name2idx = {name : i for i, name in enumerate(self.body_idx2name)}

        self.dof_idx2name = self.gym.get_asset_dof_names(robot_asset)
        print('DoF names:', self.dof_idx2name)
        self.dof_name2idx = {name : i for i, name in enumerate(self.dof_idx2name)}

        # Initial states
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
        start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)
        self.base_start_pose = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        print('Start pose:', self.base_start_pose)
        self.base_start_pose = to_torch(self.base_start_pose, dtype = torch.float, device = self.device, requires_grad = False).unsqueeze(0)

        # Set the property of the degree of freedoms
        # dof_props: ('hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature')
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        # dof_props['hasLimits'][:] = 1
        for name, i in self.dof_name2idx.items():
            default_angle = self.cfg.init_state.default_joint_angles[name]
            dof_props['lower'][i] = default_angle - self.cfg.control.action_scale * self.cfg.normalization.clip_actions * 1.2
            dof_props['upper'][i] = default_angle + self.cfg.control.action_scale * self.cfg.normalization.clip_actions * 1.2
            dof_props['stiffness'][i] = self.cfg.control.stiffness[name]
            dof_props['damping'][i] = self.cfg.control.damping[name]
        dof_props['driveMode'][:] = self.cfg.asset.dof_props.default_dof_drive_mode # 1: gymapi.DOF_MODE_POS
        dof_props['velocity'][:] = self.cfg.asset.dof_props.velocity
        dof_props['effort'][:] = self.cfg.asset.dof_props.effort
        # dof_props['friction'][:] = self.cfg.asset.dof_props.friction
        # dof_props['armature'][:] = self.cfg.asset.dof_props.armature

        print('dof_props:', dof_props)

        # Create every environment instance.
        spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.actor_handles = []
        num_envs_per_row = int(self.num_envs**0.5)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_envs_per_row)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)

            # Domain randomization
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia = True)

            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # Index the feet. Warning: setting collapse_fixed_joints to True may cause some links to disappear in body_names.
        foot_shank_names = self.cfg.asset.foot_shank_names 
        self.foot_shank_indices = torch.zeros(len(foot_shank_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(foot_shank_names)):
            self.foot_shank_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], foot_shank_names[i])
        print('Foot shank links:', list(zip(foot_shank_names, self.foot_shank_indices.tolist())))

        foot_sole_names = self.cfg.asset.foot_sole_names
        self.foot_sole_indices = torch.zeros(len(foot_sole_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(foot_sole_names)):
            self.foot_sole_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], foot_sole_names[i])
        print('Foot sole links:', list(zip(foot_sole_names, self.foot_sole_indices.tolist())))

        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
        print('Knee links:', list(zip(knee_names, self.knee_indices.tolist())))

        # Index the base.
        base_name = self.cfg.asset.base_name
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name)
        print('Base link:', (base_name, self.base_index))


    '''
        Controllers.
    '''
    def _position_control(self, actions):
        target_dof_pos = actions * self.cfg.control.action_scale + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_dof_pos))


    '''
        Step the simulation.
    '''
    def step(self, actions):
        '''
            actions: the prediction from the policy.
        '''
        self.pre_physics_step()
        self.actions[:] = actions.to(self.device)
        clip_act_range = self.cfg.normalization.clip_actions
        clip_actions = torch.clamp(actions.to(self.device), -clip_act_range, clip_act_range)
        self.render()
        self._render_cameras()

        for i in range(self.cfg.control.control_frequency):
            '''
                Position control or torque control?
            '''
            self._position_control(clip_actions)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Update the buffers
            self.gym.refresh_dof_state_tensor(self.sim)  # done in step

        self.post_physics_step()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras    


    def pre_physics_step(self):
        # Store the historical states.
        for key in self.save_history_data_keys:
            self.history_data[key].append(getattr(self, key).clone())


    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1

        # Compute the observations
        obs = self.compute_observations()
        clip_obs_range = self.cfg.normalization.clip_observations
        self.obs_buf[:] = torch.clip(obs, -clip_obs_range, clip_obs_range)

        self.E_C_frc[:], self.E_C_spd[:] = self._compute_E_C()

        # Compute the rewards after the observations.
        self.reset_buf[:], self.time_out_buf[:] = self.check_termination()
        self.rew_buf[:] = self.compute_rewards()

        self._update_foot_periodicity_visualization(env_idx = self.record_video_env_idx)

        # Reset if any environment terminates
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)


    def _push_robot(self, torques = None):
        pass


    def reset_idx(self, env_ids):
        """
            Reset the terminated environments.
        """
        if len(env_ids) == 0:
            return 
        
        # Episode information
        self.extras['episode'] = {}
        # Calculate the average reward term per time-step
        for rew_key in self.episode_rew_sums.keys():
            self.extras['episode'][f'reward_{rew_key}'] = torch.mean(self.episode_rew_sums[rew_key][env_ids] / (self.episode_length_buf[env_ids]))
            self.episode_rew_sums[rew_key][env_ids] = 0
        
        # Reset states.
        add_noise = self.cfg.init_state.noise.add_noise
        self._reset_dofs(env_ids, add_noise = add_noise)
        self._reset_root_states(env_ids, add_noise = add_noise)
        self._reset_commands(env_ids)
        self._reset_foot_periodicity(env_ids, calculate_from_slip_model = True, add_noise = self.foot_periodicity_cfg.add_noise)

        # Reset buffers.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # self.feet_air_time[env_ids] = 0


    def _reset_dofs(self, env_ids, add_noise = False):        
        self.dof_pos[env_ids] = self.default_dof_pos 
        self.dof_vel[env_ids] = 0

        if add_noise:
            self.dof_pos[env_ids] += torch_rand_float(*self.cfg.init_state.noise.dof_pos, shape = (len(env_ids), self.num_dof), device = self.device)
            self.dof_vel[env_ids] += torch_rand_float(*self.cfg.init_state.noise.dof_vel, shape = (len(env_ids), self.num_dof), device = self.device)

        env_ids_int32 = env_ids.to(dtype = torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )
        

    def _reset_root_states(self, env_ids, add_noise = False):
        self.root_states[env_ids] = self.base_start_pose

        if add_noise:
            self.root_states[env_ids, 7:10] += torch_rand_float(*self.cfg.init_state.noise.base_lin_vel, (len(env_ids), 3), device = self.device)
            self.root_states[env_ids, 10:13] += torch_rand_float(*self.cfg.init_state.noise.base_ang_vel, (len(env_ids), 3), device = self.device)

        env_ids_int32 = env_ids.to(dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )


    def _reset_commands(self, env_ids):
        # Set linear velocity commands
        base_lin_vel_min, base_lin_vel_max = self.cfg.commands.base_lin_vel_min, self.cfg.commands.base_lin_vel_max
        for idx, (min_vel, max_vel) in enumerate(zip(base_lin_vel_min, base_lin_vel_max)):
            self.command_lin_vel[env_ids, idx:idx+1] = torch_rand_float(lower = min_vel, upper = max_vel, shape = (len(env_ids), 1), device = self.device)

        # Set angular velocity commands
        base_ang_vel_min, base_ang_vel_max = self.cfg.commands.base_ang_vel_min, self.cfg.commands.base_ang_vel_max
        for idx, (min_vel, max_vel) in enumerate(zip(base_ang_vel_min, base_ang_vel_max)):
            self.command_ang_vel[env_ids, idx:idx+1] = torch_rand_float(lower = min_vel, upper = max_vel, shape = (len(env_ids), 1), device = self.device)
    

    def compute_observations(self):
        base_lin_vels = self._get_base_lin_vel(self.root_states) # 3
        base_ang_vels = self._get_base_ang_vel(self.root_states) # 3
        base_proj_grav = self._get_base_projected_gravity(self.root_states, self.gravity_vec) # 3
        cmd_lin_vels = self.command_lin_vel[..., self.cfg.commands.base_lin_vel_axis] # 2
        cmd_ang_vels = self.command_ang_vel[..., self.cfg.commands.base_ang_vel_axis] # 1
        dof_pos = (self.dof_pos - self.default_dof_pos) # 9
        dof_vel = self.dof_vel # 9
        prev_actions = self.actions # 9

        # num_feet: 4
        # phis = self._get_periodicity_ratio()
        # foot_phis_sin = torch.sin(2 * torch.pi * (phis.unsqueeze(-1) + self.foot_thetas))
        foot_phis_sin = torch.sin(2 * torch.pi * self._get_foot_phis())
                              
        # phase ratios: 2
        phase_ratios = torch.stack([
            1.0 - self.duty_factors, # swing phase ratio
            self.duty_factors, # stance phase ratio
        ], dim = -1)


        concat_obs = torch.cat([
            base_lin_vels,
            base_ang_vels,
            dof_pos,
            dof_vel,
            base_proj_grav,
            cmd_lin_vels,
            cmd_ang_vels,
            prev_actions,
            foot_phis_sin,    
            phase_ratios
        ], dim = -1)

        if self.domain_rand_cfg.observation.apply:
            concat_obs += self._process_obs_noises()

        return concat_obs
    

    def check_termination(self):
        """
            Check if the environments need to be reset.
        """
        reset = torch.zeros_like(self.episode_length_buf, dtype = torch.bool)

        # Reach the time limits.
        timeout = self.episode_length_buf >= self.max_episode_length
        reset |= timeout

        # If the robot is going to flip
        proj_grav = self._get_base_projected_gravity(self.root_states, self.gravity_vec)
        flip = torch.abs(proj_grav[..., -1]) < 0.9
        reset |= flip

        # If the robot base is below the certain height.
        base_height = self._get_base_pos(self.root_states)[..., -1]
        in_alive_height = torch.logical_or(base_height < 0.025, base_height > 0.075)
        reset |= in_alive_height

        # If any knee contacts the ground
        knee_frcs = self._get_contact_forces(self.knee_indices)
        knee_contact = torch.any(knee_frcs > 1.0, dim = -1)
        reset |= knee_contact
        
        return reset, timeout
        

    def compute_rewards(self):
        rewards = torch.zeros_like(self.episode_length_buf, dtype = torch.float)
        
        for rew_key, rew_func in self.reward_functions.items():
            curr_rew_term = rew_func()
            self.episode_rew_sums[rew_key] += curr_rew_term
            rewards += curr_rew_term

        rewards = torch.clamp(rewards, min = 0)
        self.episode_rew_sums['total'] += rewards

        return rewards
    

    '''
        Handle foot periodicity in the environment.
    '''
    def _init_foot_periodicity_buffer(self):
        foot_periodicity_cfg = self.cfg.foot_periodicity
        self.init_foot_thetas = torch.as_tensor(self.cfg.foot_periodicity.init_foot_thetas, dtype = torch.float, device = self.device).unsqueeze(0)
        self.duty_factors = torch.ones_like(self.episode_length_buf) * foot_periodicity_cfg.duty_factor # Duty factor (the ratio of the stance phase) of the current gait
        self.kappa = torch.ones_like(self.duty_factors) * foot_periodicity_cfg.kappa # Kappa
        self.foot_thetas = torch.as_tensor(foot_periodicity_cfg.init_foot_thetas, dtype = torch.float, device = self.device).unsqueeze(0).repeat(self.num_envs, 1) # Clock input shift
        self.gait_period_steps = torch.ones_like(self.episode_length_buf) * foot_periodicity_cfg.gait_period / self.dt


    def _reset_foot_periodicity(self, env_ids, calculate_from_slip_model = False, add_noise = False):
        foot_periodicity_cfg = self.cfg.foot_periodicity
        self.kappa[env_ids] = foot_periodicity_cfg.kappa
        self.foot_thetas[env_ids, :] = self.init_foot_thetas

        if calculate_from_slip_model:
            # Reset the parameters to the values calculated from the SLIP model.
            self.duty_factors[env_ids] = self._compute_duty_factor_from_cmd_forward_linvel(self.command_lin_vel[env_ids, 0])
            self.gait_period_steps[env_ids] = self._compute_period_from_cmd_forward_linvel(self.command_lin_vel[env_ids, 0]) / self.dt

            # # Enforce time-reversal symmetry if the command forward linear velocity is negative
            # negative_cmd_forward_linvel_indices = env_ids[torch.where(self.command_lin_vel[env_ids] < 0)[0]]
            # self.foot_thetas[negative_cmd_forward_linvel_indices, :] *= -1
            # self.foot_thetas[negative_cmd_forward_linvel_indices, :] += 1 - self.duty_factors[negative_cmd_forward_linvel_indices].unsqueeze(-1)
        else:
            # Reset the parameters to the values in the configuration file.
            self.duty_factors[env_ids] = foot_periodicity_cfg.duty_factor
            self.gait_period_steps[env_ids] = foot_periodicity_cfg.gait_period / self.dt

        # Randomization to foot thetas
        if add_noise:
            noise_scale = self.foot_periodicity_cfg.noise_scale
            noise_level = self.foot_periodicity_cfg.noise_level
            self.foot_thetas[env_ids, :] += torch.randint(low = -noise_level, high = noise_level, size = (len(env_ids), len(self.foot_sole_indices)), device = self.device) * noise_scale

        # print(self.foot_thetas)

        self.E_C_frc[:], self.E_C_spd[:] = self._compute_E_C()
        


    def _get_contact_forces(self, indices):
        return torch.norm(self.contact_forces[:, indices, :], dim = -1)
    

    def _get_rb_lin_vels(self, indices):
        return torch.norm(self.rigid_body_states[:, indices, 7:10], dim = -1)
    

    def _get_periodicity_ratio(self):
        return self.episode_length_buf / self.gait_period_steps
    

    def _get_foot_phis(self):
        phi = self._get_periodicity_ratio()
        sign_indicator = torch.ones_like(self.command_lin_vel[..., 0:1])
        sign = torch.where(self.command_lin_vel[..., 0:1] >= 0, sign_indicator, -sign_indicator)
        return torch.abs(((phi.unsqueeze(-1) + self.foot_thetas) * sign) % sign)

    
    def _compute_E_C(self):
        foot_periodicity_cfg = self.cfg.foot_periodicity
        # phi = self._get_periodicity_ratio().cpu().numpy()[..., None] # [num_envs, 1]
        # thetas = self.foot_thetas.cpu().numpy() # [num_envs, len(self.foot_indices)]
        foot_phis = self._get_foot_phis().cpu().numpy()
        duty_factor = self.duty_factors.cpu().numpy()[..., None] # [num_envs, 1]
        kappa = self.kappa.cpu().numpy()[..., None] # [num_envs, 1]

        E_C_frc = E_periodic_property(foot_phis, duty_factor, kappa, foot_periodicity_cfg.c_swing_frc, foot_periodicity_cfg.c_stance_frc)
        E_C_frc = torch.from_numpy(E_C_frc).to(self.device)
        E_C_spd = E_periodic_property(foot_phis, duty_factor, kappa, foot_periodicity_cfg.c_swing_spd, foot_periodicity_cfg.c_stance_spd)
        E_C_spd = torch.from_numpy(E_C_spd).to(self.device)

        return E_C_frc, E_C_spd
    

    # When recording videos, visualize the foot contact forces and velocities.
    def _init_foot_periodicity_visualization(self):
        self.record_video_env_idx = 0
        if self.record_video:
            self.foot_periodicity_vis_data = defaultdict(lambda: deque(maxlen = self.max_episode_length))
        else:
            self.foot_periodicity_vis_data = None


    def _update_foot_periodicity_visualization(self, env_idx = 0):
        if self.foot_periodicity_vis_data is not None:
            # Need to call clone() to avoid soft copy
            cmd_forward_linvel = self.command_lin_vel[env_idx, 0].item()
            self.foot_periodicity_vis_data['phi'].append( (1 if cmd_forward_linvel >= 0 else -1) * self._get_periodicity_ratio()[env_idx].clone())
            self.foot_periodicity_vis_data['True_frc'].append(self._get_contact_forces(self.foot_shank_indices)[env_idx].clone())
            self.foot_periodicity_vis_data['True_spd'].append(self._get_rb_lin_vels(self.foot_sole_indices)[env_idx].clone())
            self.foot_periodicity_vis_data['E_frc'].append(self.E_C_frc[env_idx].clone())
            self.foot_periodicity_vis_data['E_spd'].append(self.E_C_spd[env_idx].clone())
            self.foot_periodicity_vis_data['cmd_forward_linvel'].append(self.command_lin_vel[env_idx, 0].clone())
            self.foot_periodicity_vis_data['real_forward_linvel'].append(self._get_base_lin_vel(self.root_states)[env_idx, 0].clone())


    def _save_foot_periodicity_visualization(self, file_name = 'fp'):
        if self.foot_periodicity_vis_data is not None:
            for key, data in self.foot_periodicity_vis_data.items():
                self.foot_periodicity_vis_data[key] = torch.stack(list(data), dim = -1).cpu().numpy()
                # print(key, self.foot_periodicity_vis_data[key].shape)
            self.foot_periodicity_vis_data['foot_shanks'] = self.cfg.asset.foot_shank_names
            self.foot_periodicity_vis_data['foot_soles'] = self.cfg.asset.foot_sole_names
            # Save the other information. A concern is that calling reset() before the recording terminates may cause the wrong information to be saved, need to address this issue.
            self.foot_periodicity_vis_data['cmd_linvel'] = self.command_lin_vel[self.record_video_env_idx, :].tolist()
            self.foot_periodicity_vis_data['duty_factor'] = self.duty_factors[self.record_video_env_idx].item()
            with open(f'{file_name}.pkl', 'wb') as handle:
                pickle.dump(dict(self.foot_periodicity_vis_data), handle)


    '''
        Generate the gait parameters from the command linear velocity. 
    '''
    def _compute_period_from_cmd_forward_linvel(self, cmd_forward_linvel):
        abs_cmd_forward_linvel = torch.abs(cmd_forward_linvel)
        random_scale = torch.rand_like(cmd_forward_linvel) * 2 - 1.0
        return 0.2576 * torch.exp(-0.9829 * abs_cmd_forward_linvel) * (1 + random_scale * abs_cmd_forward_linvel * 0.25)
    

    def _compute_duty_factor_from_cmd_forward_linvel(self, cmd_forward_linvel):
        abs_cmd_forward_linvel = torch.abs(cmd_forward_linvel)
        random_scale = torch.rand_like(cmd_forward_linvel) * 2 - 1.0
        return 0.5588 * torch.exp(-0.6875 * abs_cmd_forward_linvel) * (1 + random_scale * abs_cmd_forward_linvel * 0.25)


    '''
        Reward functions
    '''
    def _prepare_reward_functions(self, rewards_dict : dict):                
        return {key : getattr(self, f'_reward_{key}') for key in rewards_dict.keys()}


    def _reward_alive_bonus(self):
        alive_reward = torch.zeros_like(self.rew_buf)
        coef = self.cfg.rewards.alive_bonus.coef
        return alive_reward + coef


    def _reward_track_lin_vel(self):
        # By default, consider tracking the linear velocity at x/y axis.
        axis = self.cfg.commands.base_lin_vel_axis
        lin_vel_err = torch.abs(self.command_lin_vel - self._get_base_lin_vel(self.root_states))[..., axis]
        scale = self.reward_cfg.track_lin_vel.scale
        coef = self.reward_cfg.track_lin_vel.coef
        return torch.sum(negative_exponential(lin_vel_err, scale, coef), dim = -1)


    def _reward_track_ang_vel(self):
        # By default, consider tracking the angular velocity at z axis.
        axis = self.cfg.commands.base_ang_vel_axis
        ang_vel_err = torch.abs(self.command_ang_vel - self._get_base_ang_vel(self.root_states))[..., axis]
        scale = self.cfg.rewards.track_ang_vel.scale
        coef = self.cfg.rewards.track_ang_vel.coef
        return torch.sum(negative_exponential(ang_vel_err, scale, coef), dim = -1)
    

    def _reward_torques(self):
        torque_term = torch.sum(torch.abs(self.history_data['torques'][-1] - self.torques), dim = -1)
        scale = self.cfg.rewards.torques.scale
        coef = self.cfg.rewards.torques.coef
        return negative_exponential(torque_term, scale, coef)
    

    def _reward_foot_periodicity(self):
        # Here both E_C_frc and E_C_spd are in [-1, 0], so negate them when using negative_exponential().

        foot_frcs = self._get_contact_forces(self.foot_shank_indices)
        foot_frc_scale = self.cfg.rewards.foot_periodicity.scale_frc
        foot_frc_coef = self.cfg.rewards.foot_periodicity.coef_frc
        R_E_C_frc = torch.sum(negative_exponential(foot_frcs, foot_frc_scale, foot_frc_coef * -self.E_C_frc), dim = -1)

        foot_spds = self._get_rb_lin_vels(self.foot_sole_indices)
        foot_spd_scale = self.cfg.rewards.foot_periodicity.scale_spd
        foot_spd_coef = self.cfg.rewards.foot_periodicity.coef_spd
        R_E_C_spd = torch.sum(negative_exponential(foot_spds, foot_spd_scale, foot_spd_coef * -self.E_C_spd), dim = -1)

        return R_E_C_frc + R_E_C_spd


    def _reward_morphological_symmetry(self):
        '''
            Pronking gait.
            Dof indices (s -> shoulder, t -> knee):
            'neck_joint' : 0, 
            'shlfs_joint' : 1,  lf shoulder
            'shlft_joint' : 2,  lf thigh
            'shlrs_joint' : 3,  lr shoulder
            'shlrt_joint' : 4,  lr thigh
            'shrfs_joint' : 5,  rf shoulder
            'shrft_joint' : 6,  rf thigh
            'shrrs_joint' : 7,  rr shoulder
            'shrrt_joint' : 8   rr thigh

            foot theta orders: lf, lr, rf, rr
        '''

        # lf_rf_consistent = (self.foot_thetas[..., 0] == self.foot_thetas[..., 2]).float()
        # lr_rr_consistent = (self.foot_thetas[..., 1] == self.foot_thetas[..., 3]).float()
        # lf_rr_consistent = (self.foot_thetas[..., 0] == self.foot_thetas[..., 3]).float()

        # error_sum = torch.zeros_like(self.episode_length_buf, dtype = torch.float)
        # error_sum += torch.abs(self.dof_pos[..., 1] - self.dof_pos[..., 5]) * lf_rf_consistent # lf, rf, s
        # error_sum += torch.abs(self.dof_pos[..., 3] - self.dof_pos[..., 7]) * lr_rr_consistent # lr, rr, s
        # # error_sum += torch.abs(self.dof_pos[..., 1] + self.dof_pos[..., 7]) * lf_rr_consistent # lf, rr, s
        # error_sum += torch.abs(self.dof_pos[..., 2] - self.dof_pos[..., 6]) * lf_rf_consistent # lf, rf, t
        # error_sum += torch.abs(self.dof_pos[..., 4] - self.dof_pos[..., 8]) * lr_rr_consistent # lr, rr, t
        # # error_sum += torch.abs(self.dof_pos[..., 2] - self.dof_pos[..., 8]) * lf_rr_consistent # lf, rr, t

        # scale = self.reward_cfg.morphological_symmetry.scale
        # coef = self.reward_cfg.morphological_symmetry.coef
        # return negative_exponential(error_sum, scale, coef)

        threshold = 0.01
        lf_rf_consistent = torch.abs(self.foot_thetas[..., 0] - self.foot_thetas[..., 2]) <= threshold # left-right symmetry
        lr_rr_consistent = torch.abs(self.foot_thetas[..., 1] - self.foot_thetas[..., 3]) <= threshold # left-right symmetry
        lf_lr_consistent = torch.abs(self.foot_thetas[..., 0] - self.foot_thetas[..., 1]) <= threshold # front-back symmetry 
        rf_rr_consistent = torch.abs(self.foot_thetas[..., 2] - self.foot_thetas[..., 3]) <= threshold # front-back symmetry
        lf_rr_consistent = torch.abs(self.foot_thetas[..., 0] - self.foot_thetas[..., 3]) <= threshold # diagonal symmetry 
        rf_lr_consistent = torch.abs(self.foot_thetas[..., 2] - self.foot_thetas[..., 1]) <= threshold # diagonal symmetry

        error_sum = torch.zeros_like(self.episode_length_buf, dtype = torch.float)
        error_sum += torch.abs(self.dof_pos[..., 1] - self.dof_pos[..., 5]) * lf_rf_consistent # lf, rf, shoulder left-right symmetry
        error_sum += torch.abs(self.dof_pos[..., 2] - self.dof_pos[..., 6]) * lf_rf_consistent # lf, rf, thigh left-right symmetry
        error_sum += torch.abs(self.dof_pos[..., 3] - self.dof_pos[..., 7]) * lr_rr_consistent # lr, rr, shoulder left-right symmetry
        error_sum += torch.abs(self.dof_pos[..., 4] - self.dof_pos[..., 8]) * lr_rr_consistent # lr, rr, thigh left-right symmetry
        error_sum += torch.abs(self.dof_pos[..., 1] + self.dof_pos[..., 3]) * lf_lr_consistent # lf, lr, shoulder front-rear symmetry
        error_sum += torch.abs(self.dof_pos[..., 2] - self.dof_pos[..., 4]) * lf_lr_consistent # lf, lr, thigh front-rear symmetry
        error_sum += torch.abs(self.dof_pos[..., 5] + self.dof_pos[..., 7]) * rf_rr_consistent # rf, rr, shoulder front-rear symmetry
        error_sum += torch.abs(self.dof_pos[..., 6] - self.dof_pos[..., 8]) * rf_rr_consistent # rf, rr, thigh front-rear symmetry
        error_sum += torch.abs(self.dof_pos[..., 1] + self.dof_pos[..., 7]) * lf_rr_consistent # lf, rr, shoulder diagonal symmetry
        error_sum += torch.abs(self.dof_pos[..., 2] - self.dof_pos[..., 8]) * lf_rr_consistent # lf, rr, thigh diagonal symmetry
        error_sum += torch.abs(self.dof_pos[..., 5] + self.dof_pos[..., 3]) * rf_lr_consistent # rf, lr, shoulder diagonal symmetry
        error_sum += torch.abs(self.dof_pos[..., 6] - self.dof_pos[..., 4]) * rf_lr_consistent # rf, lr, thigh diagonal symmetry

        scale = self.reward_cfg.morphological_symmetry.scale
        coef = self.reward_cfg.morphological_symmetry.coef
        return negative_exponential(error_sum, scale, coef)
    

    def _reward_pitching(self):
        # order: lf, lr, rf, rr
        # term = torch.abs(self._get_base_ang_vel(self.root_states)[..., 1]) # penalize ang vel
        term = torch.abs(self._get_base_lin_vel(self.root_states)[..., -1]) # penalize lin vel z 
        scale = self.reward_cfg.pitching.scale
        coef = self.reward_cfg.pitching.coef
        return negative_exponential(term, scale, coef)
    

    def _reward_collision(self):
        # collision_term = torch.
        pass

    

'''
    Reward templates
'''
# y = A * -(1 - exp(-B * x))
def negative_exponential(x, scale, coef):
    return -coef * (1 - torch.exp(-scale * x))


def exponential(x, scale, coef):
    return coef * torch.exp(-scale * x)


'''
    Foot periodicity using Vonmise distribution.
'''
def limit_input_vonmise_cdf(x, loc, kappa):
    # Ranges: x in [0, 1]
    # assert np.min(x) >= 0.0 and np.max(x) <= 1.0
    return vonmises_line.cdf(x = 2*np.pi*x, loc = 2*np.pi*loc, kappa = kappa)


def prob_phase_indicator(phi : np.ndarray, start : np.ndarray, end : np.ndarray, kappa : np.ndarray):
    # P(I = 1)
    # start, end, kappa: [num_envs, 1]; phi: [num_envs, num_feet]
    temp = np.stack([start, end, start - 1.0, end - 1.0, start + 1.0, end + 1.0], axis = 0) # For faster computation on vonmise cdf, shape [6, num_envs, 1]
    Ps = limit_input_vonmise_cdf(phi[None], temp, kappa[None])
    return Ps[0] * (1 - Ps[1]) + Ps[2] * (1 - Ps[3]) + Ps[4] * (1 - Ps[5])


def E_phase_indicator(phi, start, end, kappa):
    # E_I = 1 * P(I = 1) + 0 * P(I = 0)
    return prob_phase_indicator(phi, start, end, kappa)


def E_periodic_property(phi, duty_factor, kappa, c_swing, c_stance):
    return c_swing * E_phase_indicator(phi, np.zeros_like(duty_factor), 1.0 - duty_factor, kappa) + c_stance * E_phase_indicator(phi, 1.0 - duty_factor, np.ones_like(duty_factor), kappa)


'''
    Convert RGBA images to RGB images
'''
def RGBA2RGB(rgba_img : np.ndarray, rgb_background = [0, 0, 0]):
    alpha = rgba_img[..., -1]
    return np.stack([
        (1 - alpha) * rgb_background[0] + alpha * rgba_img[..., 0],
        (1 - alpha) * rgb_background[1] + alpha * rgba_img[..., 1],
        (1 - alpha) * rgb_background[2] + alpha * rgba_img[..., 2],
    ], axis = -1).astype(np.uint8)


'''
    Plot the foot contact forces and velocities.
'''
def plot_foot_periodicity(data_file_path, fig_name = 'fp'):
    with open(data_file_path, 'rb') as handle:
        data = pickle.load(handle)

    # Plot foot contact forces
    num_subplots = len(data['foot_shanks'])
    fig, axs = plt.subplots(nrows = num_subplots, sharex = True)
    fig.set_figheight(num_subplots * 2)
    fig.set_figwidth(16)
    for idx in range(num_subplots):
        color = 'tab:red'
        axs[idx].plot(data['phi'], data['E_frc'][idx], color = color)
        axs[idx].set_ylabel('E_frc', color = color)
        axs[idx].tick_params(axis = 'y', labelcolor = color)
        axs[idx].set_title(data['foot_shanks'][idx])

        twin_ax = axs[idx].twinx()
        color = 'tab:blue'
        twin_ax.set_ylabel('True_frc', color = color)
        twin_ax.plot(data['phi'], data['True_frc'][idx], color = color)
        twin_ax.tick_params(axis = 'y', labelcolor = color)

    axs[-1].set_xlabel('$\phi$')
    fig.suptitle(f'cmd_linvel = {data["cmd_linvel"]}, duty_factor = {data["duty_factor"]}')
    fig.tight_layout()
    fig.savefig(f'{fig_name}_frc.png')

    # Plot foot velocities
    num_subplots = len(data['foot_soles'])
    fig, axs = plt.subplots(nrows = num_subplots, sharex = True)
    fig.set_figheight(num_subplots * 2)
    fig.set_figwidth(16)
    for idx in range(num_subplots):
        color = 'tab:red'
        axs[idx].plot(data['phi'], data['E_spd'][idx], color = color)
        axs[idx].set_ylabel('E_spd', color = color)
        axs[idx].tick_params(axis = 'y', labelcolor = color)
        axs[idx].set_title(data['foot_soles'][idx])

        twin_ax = axs[idx].twinx()
        color = 'tab:green'
        twin_ax.set_ylabel('True_spd', color = color)
        twin_ax.plot(data['phi'], data['True_spd'][idx], color = color)
        twin_ax.tick_params(axis = 'y', labelcolor = color)

    axs[-1].set_xlabel('$\phi$')
    fig.suptitle(f'cmd_linvel = {data["cmd_linvel"]}, duty_factor = {data["duty_factor"]}')
    fig.tight_layout()
    fig.savefig(f'{fig_name}_spd.png')

    # Plot forward velocities
    fig, axs = plt.subplots(nrows = 1, ncols = 1)
    axs.plot(data['phi'], data['cmd_forward_linvel'], label = 'cmd')
    axs.plot(data['phi'], data['real_forward_linvel'], label = 'real')
    axs.set_title(f'cmd_linvel = {data["cmd_linvel"]}, duty_factor = {data["duty_factor"]}')
    axs.set_xlabel('$\phi$')
    axs.set_ylabel('base forward vel (m/s)')
    fig.legend()
    fig.savefig(f'{fig_name}_basevel.png')