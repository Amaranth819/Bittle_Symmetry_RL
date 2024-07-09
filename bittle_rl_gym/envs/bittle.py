import os
import torch
from isaacgym import gymapi, gymtorch
from bittle_rl_gym.envs.base_task import BaseTask
from typing import List



class Bittle(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


        if not self.headless:
            # Set camera
            pass


    def _set_camera(
            self, 
            pos : List[int], 
            lookat : List[int], 
            env_id : int = -1
        ):
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
        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)

        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)


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
            Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_envs_per_row):
        """
            Create #num_envs environment instances in total.
        """
        # Load the urdf and mesh files of the robot.
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
        asset_file = 'urdf/bittle.urdf'

        # Set some physical properties of the robot.
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.4
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        # Initial states
        base_init_state_list = None # A 6-dim vector read from the configuration file. [:3] is position and [3:] is rotation.
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*base_init_state_list[:3])
        start_pose.r = gymapi.Quat.from_euler_zyx(*base_init_state_list[3:])

        # Set the property of the degree of freedoms
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = None # self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"] #gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = None # self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = None # self.cfg["env"]["control"]["damping"] #self.Kd

        # Create every environment instance.
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.actor_handles = []
        for i in range(num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_envs_per_row)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "bittle", i, 1, 0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # Index the feet.
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        feet_names = [] 
        self.feet_indices = torch.zeros(len(feet_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # Index the knees.
        knee_names = []
        self.knee_indices = torch.zeros(len(knee_names), dtype = torch.long, device = self.device, requires_grad = False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        # Index the base.
        base_link_name = ""
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_link_name)


    def pre_physics_step(self, actions):
        pass


    def step(self, actions):
        pass


    def post_physics_step(self):
        pass


    def _push_robot(self, torques = None):
        pass


    def check_termination(self):
        """
            Check if the environments need to be reset.
        """
        pass


    def reset_idx(self, env_ids):
        """
            Reset the terminated environments.
        """
        if len(env_ids):
            return
        

    def compute_rewards(self):
        pass


