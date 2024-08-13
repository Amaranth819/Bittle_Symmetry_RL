from bittle_rl_gym.cfg.base_config import BaseConfig


class BittleOfficialConfig(BaseConfig):
    class env:
        num_envs = 2048
        num_observations = 42
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 9
        env_spacing = 1.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        # test = False


    class terrain:
        static_friction = 0.2 # 1.0  # [-]
        dynamic_friction = 0.2 # 1.0  # [-]
        restitution = 0.        # [-]


    # viewer camera:
    class viewer:
        ref_env = -1
        pos = [0, -0.4, 0.06]  # [m]
        lookat = [0, 0, 0.06]  # [m]


    class asset:
        file = 'urdf/Bittle_Petoi.urdf'
        name = "bittle"  # actor name
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 1 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

        # Name of some body components, used to index body state and contact force tensors
        foot_names = ['shank_lf_1', 'shank_lr_1', 'shank_rf_1', 'shank_rr_1'] 
        knee_names = ['c_thlf_1', 'c_thlr_1', 'c_thrf__1', 'c_thrr_1']
        base_name = "base_link"


    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        # use_gpu_pipeline = True

        class physx:
            num_threads = 4
            num_subscenes = 4
            solver_type = 1  # 0: pgs, 1: tgs
            # use_gpu = True
            num_position_iterations = 4
            num_velocity_iterations = 0 # 1
            contact_offset = 0.005  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.02 # [m/s]
            max_depenetration_velocity = 100.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            friction_offset_threshold = 0.005
            friction_correlation_distance = 0.005


    class control:
        # Position/velocity/torque control
        control_type = 'P' 
        auto_PD_gains = False
        # P gains: unit [N*m/rad]
        stiffness = {
            "neck_joint" : 1,

            "shlrs_joint" : 3,
            "shrrs_joint" : 3,
            "shlfs_joint" : 3,
            "shrfs_joint" : 3,

            "shlrt_joint" : 1,
            "shrrt_joint" : 1,
            "shlft_joint" : 1.2,
            "shrft_joint" : 1.2,

            # "neck_joint" : 0.001,

            # "shlrs_joint" : 0,
            # "shrrs_joint" : 0,
            # "shlfs_joint" : 0,
            # "shrfs_joint" : 0,

            # "shlrt_joint" : 0,
            # "shrrt_joint" : 0,
            # "shlft_joint" : 0,
            # "shrft_joint" : 0,
        }
        # D gains: unit [N*m/rad]
        damping = {
            "neck_joint" : 0.001, # 0,

            "shlrs_joint" : 0.001, # 0.01,
            "shrrs_joint" : 0.001, # 0.01,
            "shlfs_joint" : 0.001, # 0.005,
            "shrfs_joint" : 0.001, # 0.005,

            "shlrt_joint" : 0.001,
            "shrrt_joint" : 0.001,
            "shlft_joint" : 0.001,
            "shrft_joint" : 0.001,
        } 
        # action scale: target = action_scale * action
        action_scale = 0.5
        # Torque limit
        torque_limit = 100
        # control_frequency: Number of control action updates @ sim DT per policy DT
        control_frequency = 4


    class init_state:
        pos = [0.0, 0.0, 0.05] # x, y, z (m)
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            "neck_joint" : -0.57,

            "shlrs_joint" : 0.57,
            "shrrs_joint" : 0.57,
            "shlfs_joint" : -0.57,
            "shrfs_joint" : -0.57,

            "shlrt_joint" : 1.14,
            "shrrt_joint" : 1.14,
            "shlft_joint" : 1.14,
            "shrft_joint" : 1.14,
        }

        class noise:
            dof_pos = [0.1, 0.1]
            dof_vel = [-0.1, 0.1]
            base_lin_vel = [-0.1, 0.1]
            base_ang_vel = [-0.1, 0.1]


    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
        clip_observations = 5.0
        clip_actions = 1.0


    class foot_periodicity:
        init_foot_thetas = [0.0, 0.5, 0.0, 0.5] # Order: same as asset.foot_names
        duty_factor = 0.43
        kappa = 16
        c_swing_frc = -1
        c_swing_spd = 0
        c_stance_frc = 0
        c_stance_spd = -1


    class commands:
        base_lin_vel_axis = [0, 1]
        base_lin_vel_min = [0.1, 0.0, 0.0]
        base_lin_vel_max = [0.1, 0.0, 0.0]
        
        base_lin_ang_axis = [2]
        base_ang_vel_min = [0.0, 0.0, 0.0]
        base_ang_vel_max = [0.0, 0.0, 0.0]


    class rewards:
        class scales:
            track_lin_vel = 10.0
            track_ang_vel = 20.0
            torque_smoothness = 10.0

            # Foot periodicity
            foot_periodicity_frc = 0.2
            foot_periodicity_spd = 5

            # Foot morphological symmetry
            foot_morpho_sym_error = 5
            foot_morpho_sym = 0.3

        class coefficients:
            alive_bonus = 0.0
            track_lin_vel = 0.9
            # track_ang_vel = 0.3
            # torque_smoothness = -0.1

            # # Foot periodicity
            # foot_periodicity_frc = 0.5
            # foot_periodicity_spd = 0.5

            # # Foot morphological symmetry
            # foot_morpho_sym = 0.3