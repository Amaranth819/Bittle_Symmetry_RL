from isaacgym import gymapi, gymtorch
from bittle_rl_gym.env.base_task import BaseTask
from typing import List
from isaacgym.torch_utils import *
from bittle_rl_gym.utils.helpers import class_to_dict
from bittle_rl_gym.env.bittle_config import BittleConfig
import os
import torch


class Bittle(BaseTask):
    def __init__(self, cfg : BittleConfig, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Set camera
        if not self.headless:
            pass

        self._parse_cfg()
        self._init_buffers()
        self._parse_rewards()


    def _parse_cfg(self):
        # dt
        self.dt = self.sim_params.dt * self.cfg.control.control_frequency
        self.max_episode_length_in_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_in_s / self.dt)

        self.auto_PD_gains = self.cfg.control.auto_PD_gains


    def _set_camera(self, pos : List[int], lookat : List[int], env_id : int = -1):
        """
            Set the camera position and direction.
            Input:
                pos: (x, y, z)
                lookat: (x, y, z)
        """
        assert len(pos) == 3 and len(lookat) == 3
        cam_pos, cam_target = gymapi.Vec3(*pos), gymapi.Vec3(*lookat)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[env_id] if env_id >= 0 else None, cam_pos, cam_target)
    

    def _init_buffers(self):
        # Actor root states: [num_actors, 13] containing position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        # Degree of freedom states: positions and velocities
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_states.view(-1, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(-1, self.num_dof, 2)[..., 1]

        # Initialize dof positions
        self.default_dof_pos = torch.zeros_like(self.dof_pos)
        default_dof_pos_from_cfg = self.cfg.init_state.default_joint_angles
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = default_dof_pos_from_cfg[name]
            self.default_dof_pos[:, i] = angle

        # Net contact forces: [num_rigid_bodies, 3]
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)

        # Torques
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(torques)

        # Rigid body states
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)

        # PD gains
        self.P_gains = torch.zeros(self.num_dof, dtype = torch.float, device = self.device, requires_grad = False)
        init_P_gains = self.cfg.control.stiffness
        for dof_name, kP in init_P_gains.items():
            try:
                dof_idx = self.dof_names.index(dof_name)
                self.P_gains[dof_idx] = kP
            except ValueError:
                continue

        self.D_gains = torch.zeros(self.num_dof, dtype = torch.float, device = self.device, requires_grad = False)
        init_D_gains = self.cfg.control.damping
        for dof_name, kD in init_D_gains.items():
            try:
                dof_idx = self.dof_names.index(dof_name)
                self.D_gains[dof_idx] = kD
            except ValueError:
                continue

        # Base orientation
        self.base_quat = self.root_states[..., 3:7]
        self.base_rpy = get_euler_xyz(self.base_quat)

        # Base velocities
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[..., 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[..., 10:13])

        # Last actions
        self.actions = torch.zeros(self.num_actions, dtype = torch.float, device = self.device, requires_grad = False)

        # Gravity and projected gravity
        self.gravity_vec = to_torch(get_axis_params(-1, self.up_axis_idx), device = self.device).repeat((self.num_envs, 1))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Forward direction
        self.forward_dir = to_torch([1, 0, 0], device = self.device).repeat((self.num_envs, 1))

        # Feet air time
        self.feet_air_time = torch.zeros((self.num_envs, len(self.feet_indices)), dtype = torch.float, device = self.device, requires_grad = False)

        # Command velocities
        self.command_lin_vel = torch.zeros((self.num_envs, 3), dtype = torch.float, device = self.device, requires_grad = False)
        self.command_ang_vel = torch.zeros_like(self.command_lin_vel)

        # Clock input for feet
        self.feet_thetas = torch.zeros_like(self.feet_air_time)

        # Duty factor
        self.duty_factors = torch.zeros(self.num_envs, dtype = torch.float, device = self.device, requires_grad = False)

        # Save the properties at the last time step
        self.last_root_states = self.root_states.clone()
        self.last_dof_pos = self.dof_pos.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.last_torques = self.torques.clone()
        self.last_rigid_body_states = self.rigid_body_states.clone()
        self.last_P_gains = self.P_gains.clone()
        self.last_D_gains = self.D_gains.clone()
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()
        self.last_actions = self.actions.clone()


    def _set_P_gains(self, P_gains_vec):
        pass


    def _set_D_gains(self, D_gains_vec):
        pass


    def create_sim(self):
        """
            Create the simulation environment.
        """
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.cfg.env.num_envs)


    def _create_ground_plane(self):
        """ 
            Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs):
        """
            Create #num_envs environment instances in total.
        """
        asset_cfg = self.cfg.asset

        # Load the urdf and mesh files of the robot.
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
        asset_file = asset_cfg.file

        # Set some physical properties of the robot.
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg.default_dof_drive_mode
        asset_options.collapse_fixed_joints = asset_cfg.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = asset_cfg.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = asset_cfg.flip_visual_attachments
        asset_options.fix_base_link = asset_cfg.fix_base_link
        asset_options.density = asset_cfg.density
        asset_options.angular_damping = asset_cfg.angular_damping
        asset_options.linear_damping = asset_cfg.linear_damping
        asset_options.armature = asset_cfg.armature
        asset_options.thickness = asset_cfg.thickness
        asset_options.disable_gravity = asset_cfg.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        # Save body and dof names.
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        # Initial states
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
        # start_pose.r = gymapi.Quat.from_euler_zyx(*base_init_state_list[3:])
        start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)
        self.base_start_pose = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_start_pose = to_torch(self.base_start_pose, dtype = torch.float, device = self.device, requires_grad = False).unsqueeze(0)

        # Set the property of the degree of freedoms
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i in range(self.num_dof):
            dof_name = self.dof_names[i]
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS # self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"] #gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg.control.stiffness[dof_name] # self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg.control.damping[dof_name] # self.cfg["env"]["control"]["damping"] #self.Kd

        # Create every environment instance.
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        num_envs_per_row = int(num_envs**0.5)
        for i in range(num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_envs_per_row)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "bittle", i, 1, 0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # Index the feet.
        foot_names = self.cfg.asset.foot_names 
        self.feet_indices = torch.zeros(len(foot_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(foot_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], foot_names[i])

        # Index the knees.
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        # Index the base.
        base_name = self.cfg.asset.base_name
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name)


    def _torque_control(self, actions):
        scaled_actions = self.cfg.control.action_scale * actions
        control_type = self.cfg.control.control_type

        if control_type == 'P':
            # Position control
            torques = self.P_gains * (scaled_actions + self.default_dof_pos - self.dof_pos) - self.D_gains * (self.dof_vel - 0)
        elif control_type == 'V':
            # Velocity control
            torques = self.P_gains * (scaled_actions + self.default_dof_pos - self.dof_pos) - self.D_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == 'T':
            # Torque control
            torques = scaled_actions
        else:
            raise TypeError('Unknown control type!')
        
        torque_limit = self.cfg.control.torque_limit
        self.last_torques[:] = self.torques[:]
        self.torques[:] = torch.clip(torques, -torque_limit, torque_limit)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))


    def _position_control(self, actions):
        target_dof_pos = actions * self.cfg.control.action_scale + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_dof_pos))


    def pre_physics_step(self, actions):
        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions


    def step(self, actions):
        '''
            actions: the prediction from the policy.
        '''
        clip_act_range = self.cfg.normalization.clip_actions
        clip_actions = torch.clamp(actions, -clip_act_range, clip_act_range).to(self.actions.device)
        self.pre_physics_step(clip_actions)
        self.render()

        for i in range(self.cfg.control.control_frequency):
            '''
                Position control or torque control?
            '''
            self._position_control(clip_actions)
            self.gym.simulate(self.sim)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

        


    def post_physics_step(self):
        self.episode_length_buf += 1

        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        obs = self.compute_observations()
        clip_obs_range = self.cfg.normalization.clip_observations
        self.obs_buf[:] = torch.clip(obs, -clip_obs_range, clip_obs_range)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[:] = torch.clip(self.privileged_obs_buf[:], -clip_obs_range, clip_obs_range)

        # Compute the rewards after the observations.
        self.rew_buf[:] = self.compute_rewards(self.actions)
        self.reset_buf[:] = self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)


    def _push_robot(self, torques = None):
        pass


    def check_termination(self):
        """
            Check if the environments need to be reset.
        """
        reset = torch.ones_like(self.episode_length_buf)

        # If the base contacts the terrain.
        reset = reset | torch.norm(self.net_contact_forces[:, self.base_index, :], dim = -1) > 1.5
        # If the knees contact the terrain.
        reset = reset | torch.any(torch.norm(self.net_contact_forces[:, self.knee_indices, :], dim = -1) > 1, dim = -1)
        # If the roll / pitch angle is beyond the threshold.
        reset = reset | torch.logical_or(torch.abs(self.base_rpy[..., 0]) > 0.8, torch.abs(self.base_rpy[..., 1] > 1.0))
        # If reaches the time limits.
        reset = reset | self.episode_length_buf >= self.max_episode_length  
        
        return reset


    def reset_idx(self, env_ids):
        """
            Reset the terminated environments.
        """
        if len(env_ids):
            return
        
        # Reset states.
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._reset_foot_periodicity(env_ids)
        
        # Reset buffers.
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0.0
        self.reset_buf[env_ids] = 1


    def _reset_dofs(self, env_ids, add_noise = True):
        if add_noise:
            dof_pos_noise = torch_rand_float(lower = 0.5, upper = 1.5, shape = (len(env_ids), self.num_dof), device = self.device)
            dof_vel_noise = torch_rand_float(lower = -0.1, upper = 0.1, shape = (len(env_ids), self.num_dof), device = self.device)
        else:
            dof_pos_noise = 1.0
            dof_vel_noise = 1.0
        
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * dof_pos_noise
        self.dof_vel[env_ids] = dof_vel_noise

        env_ids_int32 = env_ids.to(dtype = torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )
        

    def _reset_root_states(self, env_ids, add_noise = True):
        self.root_states[env_ids] = self.base_start_pose

        if add_noise:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device = self.device)

        env_ids_int32 = env_ids.to(dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )


    def _reset_foot_periodicity(self, env_ids):
        pass



    def compute_observations(self):
        self.obs_scales = self.cfg.normalization.obs_scales

        # Base height: 1
        base_height = self.root_states[..., 2].unsqueeze(-1)

        # Base linver velocity: 3
        base_lin_vel = self.base_lin_vel * self.obs_scales.lin_vel

        # Base angular velocity: 3
        base_ang_vel = self.base_ang_vel * self.obs_scales.ang_vel

        # Joint positions: 8
        dof_pos = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos

        # Joint velocities: 8
        dof_vel = self.dof_vel * self.obs_scales.dof_vel

        # Command linear velocities: 3
        command_lin_vel = self.command_lin_vel * self.obs_scales.lin_vel

        # Command angular velocities: 3
        command_ang_vel = self.command_ang_vel * self.obs_scales.ang_vel

        # Clock input of feet: len(feet_indices) = 4
        phi = self.episode_length_buf / self.max_episode_length
        feet_phis = phi.unsqueeze(-1) + self.feet_thetas
        feet_phis = torch.sin(2 * torch.pi * feet_phis)

        # Duty factor: 1
        duty_factor = self.duty_factors.unsqueeze(-1)

        # Concatenate the vectors to obtain the complete observation.
        # Dimensionality: 1 + 3 + 3 + 8 + 8 + 3 + 3 + 4 + 1 = 30
        observation = torch.cat([
            base_height,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
            command_lin_vel,
            command_ang_vel,
            feet_phis,
            duty_factor,
        ], dim = -1)

        return observation
    

    def _parse_rewards(self):
        pass

        

    def compute_rewards(self):
        rewards = torch.zeros_like(self.episode_length_buf)
        # rewards += self._reward_track_lin_vel()
        return rewards


    '''
        Reward templates
    '''
    # y = A * -(1 - exp(-B * x))
    def _rew_temp_negative_exponential(self, x, scale, coef):
        return coef * -(1 - torch.exp(-scale * x))



    def _reward_track_lin_vel(self, axis = [0, 1, 2]):
        lin_vel_err = torch.sum(torch.square(self.command_lin_vel[..., axis] - self.base_lin_vel[..., axis]), dim = -1)
        scale = self.cfg.rewards.scales.track_lin_vel
        coef = self.cfg.rewards.coefficients.track_lin_vel
        return self._rew_temp_negative_exponential(lin_vel_err, scale, coef)
    

    def _reward_track_ang_vel(self, axis = [0, 1, 2]):
        ang_vel_err = torch.sum(torch.square(self.command_ang_vel[..., axis] - self.base_ang_vel[..., axis]), dim = -1)
        scale = self.cfg.rewards.scales.track_ang_vel
        coef = self.cfg.rewards.coefficients.track_ang_vel
        return self._rew_temp_negative_exponential(ang_vel_err, scale, coef)
    

    