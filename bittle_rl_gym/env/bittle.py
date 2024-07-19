from isaacgym import gymapi, gymtorch
from bittle_rl_gym.env.base_task import BaseTask
from typing import List
from isaacgym.torch_utils import *
from bittle_rl_gym.utils.helpers import class_to_dict
from bittle_rl_gym.env.bittle_config import BittleConfig
import os
import torch
import matplotlib.pyplot as plt
from scipy.stats import vonmises_line

import time


class Bittle(BaseTask):
    def __init__(self, cfg : BittleConfig, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self._parse_cfg()
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._init_buffers()
        self._init_foot_periodicity_buffer()

        # Set camera
        if not self.headless:
            self._set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat, self.cfg.viewer.ref_env)


    def _parse_cfg(self):
        self.dt = self.sim_params.dt * self.cfg.control.control_frequency
        self.max_episode_length_in_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_in_s / self.dt)
        self.auto_PD_gains = self.cfg.control.auto_PD_gains


    def _set_camera(self, pos : List[int], lookat : List[int], ref_env_idx : int = -1):
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
            cam_target = cam_pos + ref_env_base_pos
        else:
            env_handle = None
        self.gym.viewer_camera_look_at(self.viewer, env_handle, cam_pos, cam_target)
    

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
        self.default_dof_pos = torch.zeros((self.num_dof), dtype = torch.float, device = self.device, requires_grad = False)
        default_dof_pos_from_cfg = self.cfg.init_state.default_joint_angles
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = default_dof_pos_from_cfg[name]
            self.default_dof_pos[i] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Net contact forces: [num_rigid_bodies, 3]
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # [num_envs, num_rigid_bodies, 3]

        # Torques
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(torques)

        # Rigid body states
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state) # [num_envs, num_rigid_bodies, 13]

        # Base states
        self.base_quat, self.base_rpy, self.base_lin_vel, self.base_ang_vel = self._get_base_states(self.root_states)

        # PD gains
        self.P_gains = self.cfg.control.stiffness * torch.ones(self.num_dof, dtype = torch.float, device = self.device, requires_grad = False)
        self.D_gains = self.cfg.control.damping * torch.ones(self.num_dof, dtype = torch.float, device = self.device, requires_grad = False)

        # Last actions
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype = torch.float, device = self.device, requires_grad = False)

        # Gravity and projected gravity
        self.gravity_vec = to_torch(get_axis_params(-1, self.up_axis_idx), device = self.device).repeat((self.num_envs, 1))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # # Forward direction
        # self.forward_dir = to_torch([1, 0, 0], device = self.device).repeat((self.num_envs, 1))

        # # Feet air time
        # self.foot_air_time = torch.zeros((self.num_envs, len(self.foot_indices)), dtype = torch.float, device = self.device, requires_grad = False)

        # Command velocities
        self.command_lin_vel = torch.zeros((self.num_envs, 3), dtype = torch.float, device = self.device, requires_grad = False)
        self.command_ang_vel = torch.zeros_like(self.command_lin_vel)

        # Save the properties at the last time step
        self.last_root_states = self.root_states.clone()
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()
        self.last_dof_pos = self.dof_pos.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.last_torques = self.torques.clone()
        self.last_rigid_body_states = self.rigid_body_states.clone()
        self.last_actions = self.actions.clone()
        self.last_P_gains = self.P_gains.clone()
        self.last_D_gains = self.D_gains.clone()


    def _get_base_states(self, root_states):
        base_quat = root_states[..., 3:7]
        base_rpy = torch.stack(get_euler_xyz(base_quat), dim = -1)
        base_lin_vel = quat_rotate_inverse(base_quat, root_states[..., 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, root_states[..., 10:13])
        return base_quat, base_rpy, base_lin_vel, base_ang_vel


    def _update_PD_gains(self, new_P_gains, new_D_gains):
        if self.auto_PD_gains:
            for i in range(self.num_envs):
                dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
                dof_props['stiffness'][:] = new_P_gains
                dof_props['damping'][:] = new_D_gains
                self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], dof_props)

    
    def _print_PD_gains(self):
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        print(dof_prop['stiffness'], type(dof_prop['stiffness']))
        print(dof_prop['damping'], type(dof_prop['damping']))


    def create_sim(self):
        """
            Create the simulation environment.
        """
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
        dof_props['driveMode'][:] = self.cfg.asset.default_dof_drive_mode # 1: gymapi.DOF_MODE_POS
        dof_props['stiffness'][:] = self.cfg.control.stiffness # self.Kp
        dof_props['damping'][:] = self.cfg.control.damping # Kd

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
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # Index the feet.
        foot_names = self.cfg.asset.foot_names 
        self.foot_indices = torch.zeros(len(foot_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(foot_names)):
            self.foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], foot_names[i])

        # Index the knees.
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        # Index the base.
        base_name = self.cfg.asset.base_name
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name)


    # def _torque_control(self, actions):
    #     scaled_actions = self.cfg.control.action_scale * actions
    #     control_type = self.cfg.control.control_type

    #     if control_type == 'P':
    #         # Position control
    #         torques = self.P_gains * (scaled_actions + self.default_dof_pos - self.dof_pos) - self.D_gains * (self.dof_vel - 0)
    #     elif control_type == 'V':
    #         # Velocity control
    #         torques = self.P_gains * (scaled_actions + self.default_dof_pos - self.dof_pos) - self.D_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
    #     elif control_type == 'T':
    #         # Torque control
    #         torques = scaled_actions
    #     else:
    #         raise TypeError('Unknown control type!')
        
    #     torque_limit = self.cfg.control.torque_limit
    #     self.last_torques[:] = self.torques[:]
    #     self.torques[:] = torch.clip(torques, -torque_limit, torque_limit)
    #     self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))


    def _position_control(self, actions):
        target_dof_pos = actions * self.cfg.control.action_scale + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_dof_pos))


    def pre_physics_step(self, actions):
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_states[:] = self.root_states[:]
        self.last_rigid_body_states[:] = self.rigid_body_states[:]
        self.last_torques[:] = self.torques[:]
        self.last_P_gains[:] = self.P_gains[:]
        self.last_D_gains[:] = self.D_gains[:]
        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions[:]


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
            # self._torque_control(actions)
            self.gym.simulate(self.sim)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

        self.post_physics_step()

        if not self.headless:
            self._set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat, self.cfg.viewer.ref_env)

        # return clipped obs, clipped states (None), rewards, dones and infos
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

        


    def post_physics_step(self):
        self.episode_length_buf += 1

        # Update the buffers
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Update other data
        self.base_quat[:], self.base_rpy[:], self.base_lin_vel[:], self.base_ang_vel[:] = self._get_base_states(self.root_states)

        # Compute the observations
        obs = self.compute_observations()
        clip_obs_range = self.cfg.normalization.clip_observations
        self.obs_buf[:] = torch.clip(obs, -clip_obs_range, clip_obs_range)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[:] = torch.clip(self.privileged_obs_buf[:], -clip_obs_range, clip_obs_range)

        # Compute the rewards after the observations.
        self.reset_buf[:], self.time_out_buf[:] = self.check_termination()
        self.rew_buf[:] = self.compute_rewards()

        # Reset if any environment terminates
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)


    def _push_robot(self, torques = None):
        pass


    def check_termination(self):
        """
            Check if the environments need to be reset.
        """
        reset = torch.zeros_like(self.episode_length_buf)

        # # If the base contacts the terrain. (not working normally)
        # base_contact = torch.any(torch.norm(self.contact_forces[:, self.base_index, :], dim = -1) > 1)
        # reset |= base_contact

        # # If the knees contact the terrain. (not working normally)
        # knee_contact = torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim = -1) > 5, dim = -1)
        # reset |= knee_contact

        # If the roll angle is beyond the threshold.
        roll_beyond_threshold = torch.abs(self.base_rpy[..., 0]) > 0.8
        pitch_beyond_threshold = torch.abs(self.base_rpy[..., 1]) > 1.0
        reset |= torch.logical_or(roll_beyond_threshold, pitch_beyond_threshold)

        # If reaches the time limits.
        timeout = self.episode_length_buf > self.max_episode_length  
        reset |= timeout
        
        return reset, timeout


    def reset_idx(self, env_ids):
        """
            Reset the terminated environments.
        """
        if len(env_ids) == 0:
            return 
        
        # Reset states.
        self._reset_dofs(env_ids, add_noise = False)
        self._reset_root_states(env_ids, add_noise = False)
        self._reset_commands(env_ids, add_noise = False)
        self._reset_foot_periodicity(env_ids, add_noise = False)
        
        # Reset buffers.
        self.last_actions[env_ids] = 0
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def _reset_dofs(self, env_ids, add_noise = False):
        if add_noise:
            dof_pos_noise = torch_rand_float(lower = 0.5, upper = 1.5, shape = (len(env_ids), self.num_dof), device = self.device)
            dof_vel_noise = torch_rand_float(lower = -0.1, upper = 0.1, shape = (len(env_ids), self.num_dof), device = self.device)
        else:
            dof_pos_noise = 1.0
            dof_vel_noise = 1.0
        
        self.dof_pos[env_ids] = self.default_dof_pos * dof_pos_noise
        self.dof_vel[env_ids] = dof_vel_noise

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
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device = self.device)

        env_ids_int32 = env_ids.to(dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )


    def _reset_commands(self, env_ids, add_noise = False):
        self._reset_base_lin_vel_commands(env_ids, add_noise)
        self._reset_base_ang_vel_commands(env_ids, add_noise)


    def _reset_base_lin_vel_commands(self, env_ids, add_noise = False):
        base_lin_vel_min, base_lin_vel_max = self.cfg.commands.base_lin_vel_min, self.cfg.commands.base_lin_vel_max
        command_at_all_axis = []
        for min_vel, max_vel in zip(base_lin_vel_min, base_lin_vel_max):
            command_at_all_axis.append(torch_rand_float(min_vel, max_vel, shape = (len(env_ids),), device = self.device))
        command_at_all_axis = torch.stack(command_at_all_axis, dim = -1)

        if add_noise:
            pass

        self.command_lin_vel[:] = command_at_all_axis[:]


    def _reset_base_ang_vel_commands(self, env_ids, add_noise = False):
        base_ang_vel_min, base_ang_vel_max = self.cfg.commands.base_ang_vel_min, self.cfg.commands.base_ang_vel_max
        command_at_all_axis = []
        for min_vel, max_vel in zip(base_ang_vel_min, base_ang_vel_max):
            command_at_all_axis.append(torch_rand_float(min_vel, max_vel, shape = (len(env_ids),), device = self.device))
        command_at_all_axis = torch.stack(command_at_all_axis, dim = -1)

        if add_noise:
            pass

        self.command_ang_vel[:] = command_at_all_axis[:]

        

    def _reset_foot_periodicity(self, env_ids, add_noise = False):
        pass



    # Observation
    def compute_observations(self):
        self.obs_scales = self.cfg.normalization.obs_scales

        # Base height: 1
        base_height = self.root_states[..., [2]]

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
        phi = self._get_periodicity_ratio()
        foot_phis = phi.unsqueeze(-1) + self.foot_thetas
        foot_phis = torch.sin(2 * torch.pi * foot_phis)

        # Duty factor: 1
        duty_factor = self.duty_factors.unsqueeze(-1)

        # Concatenate the vectors to obtain the complete observation.
        # Dimensionality: 1 + 3 + 3 + 8 + 8 + 3 + 3 + 4 + 1 = 34
        observation = torch.cat([
            base_height,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
            command_lin_vel,
            command_ang_vel,
            foot_phis,
            duty_factor,
        ], dim = -1)

        return observation
        

    def compute_rewards(self):
        rewards = torch.zeros_like(self.episode_length_buf)

        # Alive reward
        rewards += self._reward_alive()

        # Track linear velocity reward
        rewards += self._reward_track_lin_vel(self.cfg.commands.base_lin_vel_axis)

        # Track angular velocity reward
        rewards += self._reward_track_ang_vel(self.cfg.commands.base_lin_ang_axis)

        # Torque smoothness reward
        rewards += self._reward_torque_smoothness()

        # # Foot periodicity reward
        # rewards += self._reward_foot_periodicity()

        # Morphological symmetry reward
        rewards += self._reward_foot_morpho_symmetry(self.foot_indices[0], self.foot_indices[1]) # LF / LB
        rewards += self._reward_foot_morpho_symmetry(self.foot_indices[2], self.foot_indices[3]) # RF / RB
        rewards += self._reward_foot_morpho_symmetry(self.foot_indices[0], self.foot_indices[3]) # LF / RB
        rewards += self._reward_foot_morpho_symmetry(self.foot_indices[2], self.foot_indices[1]) # RF / LB

        return rewards


    '''
        Reward functions
    '''
    def _reward_alive(self):
        alive_reward = torch.zeros_like(self.reset_buf)
        coef = self.cfg.rewards.coefficients.alive_bonus
        return torch.where(self.reset_buf, alive_reward, alive_reward + coef)


    def _reward_track_lin_vel(self, axis = [0, 1]):
        # By default, consider tracking the linear velocity at x/y axis.
        lin_vel_err = torch.sum(torch.square(self.command_lin_vel[..., axis] - self.base_lin_vel[..., axis]), dim = -1)
        scale = self.cfg.rewards.scales.track_lin_vel
        coef = self.cfg.rewards.coefficients.track_lin_vel
        return negative_exponential(lin_vel_err, scale, coef)
    

    def _reward_track_ang_vel(self, axis = [2]):
        # By default, consider tracking the angular velocity at z axis.
        ang_vel_err = torch.sum(torch.square(self.command_ang_vel[..., axis] - self.base_ang_vel[..., axis]), dim = -1)
        scale = self.cfg.rewards.scales.track_ang_vel
        coef = self.cfg.rewards.coefficients.track_ang_vel
        return negative_exponential(ang_vel_err, scale, coef)
    

    def _reward_torque_smoothness(self):
        torque_diff = torch.sum(torch.abs(self.last_torques - self.torques), dim = -1)
        scale = self.cfg.rewards.scales.torque_smoothness
        coef = self.cfg.rewards.coefficients.torque_smoothness
        return negative_exponential(torque_diff, scale, coef)
    

    def _reward_foot_periodicity(self):
        # Here both E_C_frc and E_C_spd are in [-1, 0], so the reward is equivariant to negative_exponential() where coef is non-negative.
        E_C_frc, E_C_spd = self._compute_E_C()

        foot_frcs = self._get_contact_forces(self.foot_indices)
        foot_frc_scale = self.cfg.rewards.scales.foot_periodicity_frc
        foot_frc_coef = self.cfg.rewards.coefficients.foot_periodicity_frc        
        R_E_C_frc = foot_frc_coef * torch.sum(E_C_frc * (1 - torch.exp(-foot_frc_scale * foot_frcs)), dim = -1)

        foot_spds = self._get_lin_vels(self.foot_indices)
        foot_spd_scale = self.cfg.rewards.scales.foot_periodicity_spd
        foot_spd_coef = self.cfg.rewards.coefficients.foot_periodicity_spd
        R_E_C_spd = foot_spd_coef * torch.sum(E_C_spd * (1 - torch.exp(-foot_spd_scale * foot_spds)), dim = -1)

        return R_E_C_frc + R_E_C_spd
    

    def _reward_foot_morpho_symmetry(self, foot1_idx, foot2_idx, flipped = False):
        # If two dof_pos are flipped, then add then to calculate the difference instead.
        flip = -1 if flipped else 1
        error_scale = self.cfg.rewards.scales.foot_morpho_sym_error
        kine_error = torch.abs(self.dof_pos[..., foot1_idx] - flip * self.dof_pos[..., foot2_idx])
        scaled_kine_error = torch.exp(-0.5 * error_scale * kine_error)

        same_thetas = self.foot_thetas[..., foot1_idx] == self.foot_thetas[..., foot2_idx]
        scale = self.cfg.rewards.scales.foot_morpho_sym
        coef = self.cfg.rewards.coefficients.foot_morpho_sym

        return negative_exponential(scaled_kine_error, scale, same_thetas * coef)
    

    def _reward_pitching_vel(self):
        pass



    '''
        Handle foot periodicity in the environment.
    '''
    def _init_foot_periodicity_buffer(self):
        # Duty factor (the ratio of the stance phase) of the current gait
        self.duty_factors = torch.ones_like(self.episode_length_buf)

        # Kappa
        self.kappa = torch.ones_like(self.duty_factors)

        # Clock input shift
        self.foot_thetas = torch.zeros((self.num_envs, len(self.foot_indices)), dtype = torch.float, device = self.device, requires_grad = False)

        # Load from configuration
        self._read_foot_periodicity_from_cfg()


    def _read_foot_periodicity_from_cfg(self):
        foot_periodicity_cfg = self.cfg.foot_periodicity

        self.duty_factors[:] = foot_periodicity_cfg.duty_factor
        self.kappa[:] = foot_periodicity_cfg.kappa

        for i, val in enumerate(foot_periodicity_cfg.init_foot_thetas):
            self.foot_thetas[..., i] = val


    def _get_contact_forces(self, indices):
        return torch.norm(self.contact_forces[:, indices, :], dim = -1)
    

    def _get_lin_vels(self, indices):
        return torch.norm(self.rigid_body_states[:, indices, 7:10], dim = -1)
    

    def _get_periodicity_ratio(self):
        return self.episode_length_buf / self.max_episode_length 
    

    def _compute_E_C(self):
        foot_periodicity_cfg = self.cfg.foot_periodicity
        phi = self._get_periodicity_ratio().cpu().numpy()[..., None] # [num_envs, 1]
        duty_factor = self.duty_factors.cpu().numpy()[..., None] # [num_envs, 1]
        thetas = self.foot_thetas.cpu().numpy() # [num_envs, len(self.foot_indices)]
        kappa = self.kappa.cpu().numpy()[..., None] # [num_envs, 1]

        E_C_frc = expectation_periodic_property(phi, duty_factor, kappa, foot_periodicity_cfg.c_swing_frc, foot_periodicity_cfg.c_stance_frc, thetas)
        E_C_frc = torch.from_numpy(E_C_frc).to(self.device)

        E_C_spd = expectation_periodic_property(phi, duty_factor, kappa, foot_periodicity_cfg.c_swing_spd, foot_periodicity_cfg.c_stance_spd, thetas)
        E_C_spd = torch.from_numpy(E_C_spd).to(self.device)

        return E_C_frc, E_C_spd
    

'''
    Reward templates
'''
# y = A * -(1 - exp(-B * x))
def negative_exponential(x, scale, coef):
    return coef * -(1 - torch.exp(-scale * x))


'''
    Foot periodicity using Vonmise distribution.
'''
def limited_vonmise_cdf(x, loc, kappa):
    # Ranges: x in [0, 1], loc in [0, 1]
    # assert np.min(x) >= 0.0 and np.max(x) <= 1.0 and np.min(loc) >= 0.0 and np.max(loc) <= 1.0
    return vonmises_line.cdf(x = 2*np.pi*x, loc = 2*np.pi*loc, kappa = kappa)


def prob_phase_indicator(r, start, end, kappa, shift = 0):
    # P(I = 1)
    phi = (r + shift) % 1.0
    temp = np.stack([start, end, start - 1.0, end - 1.0, start + 1.0, end + 1.0], axis = 0)    
    Ps = limited_vonmise_cdf(phi[None], temp, kappa[None])
    return Ps[0] * (1 - Ps[1]) + Ps[2] * (1 - Ps[3]) + Ps[4] * (1 - Ps[5])


def expectation_phase_indicator(r, start, end, kappa, shift = 0):
    # E_I = 1 * P(I = 1) + 0 * P(I = 0)
    return prob_phase_indicator(r, start, end, kappa, shift)


def expectation_periodic_property(r, duty_factor, kappa, c_swing, c_stance, shift = 0):
    swing_end_ratio = 1.0 - duty_factor
    starts = np.stack([np.zeros_like(duty_factor), swing_end_ratio], axis = 0)
    ends = np.stack([swing_end_ratio, np.ones_like(duty_factor)], axis = 0)
    E_Is = expectation_phase_indicator(r, starts, ends, kappa, shift)
    return c_swing * E_Is[0] + c_stance * E_Is[1]


def test_foot_periodicity():
    num = 100
    x = np.linspace(0, 1, num)
    kappa = np.ones(num) * 16

    E_C_frcs = expectation_periodic_property(x, 0.5, kappa, -1, 0)
    plt.plot(x, E_C_frcs, '--', color = 'blue', label = 'frc')
    E_C_spds = expectation_periodic_property(x, 0.5, kappa, 0, -1)
    plt.plot(x, E_C_spds, color = 'red', label = 'spd')
    plt.legend()
    plt.savefig('E_C.png')
    plt.close()


'''
    Deprecated implementation of foot periodicity using Vonmise distribution.
'''
# def E_I_swing_frc(r, duty_factor, kappa, shift = 0):
#     return E_I(r, 0, 1.0 - duty_factor, kappa, shift)


# def E_I_stance_frc(r, duty_factor, kappa, shift = 0):
#     return E_I(r, 1.0 - duty_factor, 1.0, kappa, shift)


# def E_C_frc(r, duty_factor, kappa, c_swing_frc, c_stance_frc, shift = 0):
#     return c_swing_frc * E_I_swing_frc(r, duty_factor, kappa, shift) + c_stance_frc * E_I_stance_frc(r, duty_factor, kappa, shift)


# def E_I_swing_spd(r, duty_factor, kappa, shift = 0):
#     return E_I(r, 0, 1.0 - duty_factor, kappa, shift)


# def E_I_stance_spd(r, duty_factor, kappa, shift = 0):
#     return E_I(r, 1.0 - duty_factor, 1.0, kappa, shift)


# def E_C_spd(r, duty_factor, kappa, c_swing_spd, c_stance_spd, shift = 0):
#     return c_swing_spd * E_I_swing_spd(r, duty_factor, kappa, shift) + c_stance_spd * E_I_stance_spd(r, duty_factor, kappa, shift)