from unitree_rl_gym.legged_gym.envs.base.base_config import BaseConfig


class BittleConfig(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 48
        # num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 4.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        # test = False


    class plane:
        staticFriction: 1.0  # [-]
        dynamicFriction: 1.0  # [-]
        restitution: 0.        # [-]


    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]


    class asset:
        file = 'urdf/bittle.urdf'
        name = "bittle"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 1 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01


    class sim:
        dt =  0.005
        substeps = 2
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.002  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


    class control:
        # Position/velocity/torque control
        control_type = 'P' 
        # P gains: unit [N*m/rad]
        stiffness = { 
            "left-back-shoulder-joint" : 0.85,
            "right-back-shoulder-joint" : 0.85,
            "left-front-shoulder-joint" : 0.85,
            "right-front-shoulder-joint" : 0.85,

            "left-back-knee-joint" : 0.85,
            "right-back-knee-joint" : 0.85,
            "left-front-knee-joint" : 0.85,
            "right-front-knee-joint" : 0.85,
        }  
        # D gains: unit [N*m/rad]
        damping = {
            "left-back-shoulder-joint" : 0.04,
            "right-back-shoulder-joint" : 0.04,
            "left-front-shoulder-joint" : 0.04,
            "right-front-shoulder-joint" : 0.04,

            "left-back-knee-joint" : 0.04,
            "right-back-knee-joint" : 0.04,
            "left-front-knee-joint" : 0.04,
            "right-front-knee-joint" : 0.04,
        }     
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        controlFrequencyInv: 1 # 60 Hz


    class init_state:
        pos = [0.0, 0.0, 0.098] # x, y, z (m)
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            "left-back-shoulder-joint" : 0.52360,
            "right-back-shoulder-joint" : -0.52360,
            "left-front-shoulder-joint" : 0.52360,
            "right-front-shoulder-joint" : -0.52360,

            "left-back-knee-joint" : -1.04720,
            "right-back-knee-joint" : 1.04720,
            "left-front-knee-joint" : -1.04720,
            "right-front-knee-joint" : 1.04720,
        }


    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 5.0
        clip_actions = 1.0


    class periods:
        init_thetas = []
        duty_factor = 0.43


    class commands:
        base_lin_vel = [0.4, 0.0, 0.0]
        base_ang_vel = [0.0, 0.0, 0.0]


    class rewards:
        class reward_scales:
            pass