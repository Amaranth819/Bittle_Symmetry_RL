from bittle_rl_gym.cfg.base_config import BaseConfig


class BittleOfficialConfig(BaseConfig):
    class env:
        num_envs = 1024
        num_observations = 45
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 9
        env_spacing = 1.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 5 # episode length in seconds
        # test = False


    class terrain:
        static_friction = 1.0  # [-]
        dynamic_friction = 1.0  # [-]
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
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.4
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

        # Name of some body components, used to index body state and contact force tensors
        foot_shank_names = ['shank_lf_1', 'shank_lr_1', 'shank_rf_1', 'shank_rr_1'] 
        foot_sole_names = ['lf-foot-sole-link', 'lr-foot-sole-link', 'rf-foot-sole-link', 'rr-foot-sole-link'] 
        knee_names = ['c_thlf_1', 'c_thlr_1', 'c_thrf__1', 'c_thrr_1']
        base_name = "base_link"

        class dof_props:
            default_dof_drive_mode = 1 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
            velocity = 1.5708
            effort = 15
            # friction = 0.0
            # armature = 0.0

    class sim:
        dt =  0.005
        substeps = 2
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        # use_gpu_pipeline = True

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            # use_gpu = True
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.002  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.02 # [m/s]
            max_depenetration_velocity = 100.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            friction_offset_threshold = 0.005
            friction_correlation_distance = 0.005


    class domain_rand:
        class observation:
            apply = False
            lin_vel_noise = 0.02
            ang_vel_noise = 0.1
            dof_pos_noise = 0.0175
            dof_vel_noise = 0.1

        class rigid_shape_prop:
            apply = False
            friction_scale = [0.5, 1.5]

        class rigid_body_prop:
            apply = False
            mass_scale = [0.8, 1.2]


    class control:
        # Position/velocity/torque control
        auto_PD_gains = False
        # P gains: unit [N*m/rad]
        stiffness = {
            "neck_joint" : 1,

            "shlrs_joint" : 1,
            "shrrs_joint" : 1,
            "shlfs_joint" : 1,
            "shrfs_joint" : 1,

            "shlrt_joint" : 1,
            "shrrt_joint" : 1,
            "shlft_joint" : 1,
            "shrft_joint" : 1,
        }
        # D gains: unit [N*m/rad]
        damping = {
            "neck_joint" : 0.01, # 0,

            "shlrs_joint" : 0.01, # 0.01,
            "shrrs_joint" : 0.01, # 0.01,
            "shlfs_joint" : 0.01, # 0.005,
            "shrfs_joint" : 0.01, # 0.005,

            "shlrt_joint" : 0.01,
            "shrrt_joint" : 0.01,
            "shlft_joint" : 0.01,
            "shrft_joint" : 0.01,
        } 
        # action scale: target = action_scale * action
        action_scale = 0.5
        # Torque limit
        torque_limit = 100
        # control_frequency: Number of control action updates @ sim DT per policy DT
        control_frequency = 1


    class init_state:
        pos = [0.0, 0.0, 0.06] # x, y, z (m)
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 2024.8.16
            "neck_joint" : -0.57,

            "shlrs_joint" : 0.7854,
            "shrrs_joint" : 0.7854,
            "shlfs_joint" : -0.7854,
            "shrfs_joint" : -0.7854,

            "shlrt_joint" : 1.571,
            "shrrt_joint" : 1.571,
            "shlft_joint" : 1.571,
            "shrft_joint" : 1.571,
        }

        class noise:
            add_noise = True
            dof_pos = [0.05, 0.05]
            dof_vel = [-0.05, 0.05]
            base_lin_vel = [-0.05, 0.05]
            base_ang_vel = [-0.05, 0.05]


    class normalization:
        clip_observations = 5.0
        clip_actions = 1.0


    class foot_periodicity:
        gait_period = 0.45
        duty_factor = 0.37
        # Order: same as asset.foot_sole_names: lf, lr, rf, rr
        # init_foot_thetas = [-0.10, 0.6, 0.10, 0.4] # Galloping
        init_foot_thetas = [0.1, 0.5, -0.1, 0.5]
        # init_foot_thetas = [0.0, 0.5, 0.0, 0.5] # bounding
        kappa = 16
        c_swing_frc = -1
        c_swing_spd = 0
        c_stance_frc = 0
        c_stance_spd = -1

        add_noise = False
        noise_scale = 0.01
        noise_level = 10


    class commands:
        base_lin_vel_axis = [0, 1]
        base_lin_vel_min = [0.1, 0.0, 0.0]
        base_lin_vel_max = [0.3, 0.0, 0.0]
        
        base_ang_vel_axis = [2]
        base_ang_vel_min = [0.0, 0.0, 0.0]
        base_ang_vel_max = [0.0, 0.0, 0.0]


    class rewards:
        class alive_bonus:
            coef = 1.0
        
        class track_lin_vel:
            scale = 10.0
            coef = 0.3

        class track_ang_vel:
            scale = 5.0
            coef = 0.15

        class torques:
            scale = 0.4
            coef = 0.05

        class foot_periodicity:
            scale_frc = 1.0
            scale_spd = 5.0
            coef_frc = 0.4
            coef_spd = 0.4

        class pitching:
            scale = 8.0
            coef = 0.1

        class morphological_symmetry:
            scale = 5.0
            coef = 0.15

        # class feet_air_time:
        #     scale = 1.0
        #     coef = 0.15