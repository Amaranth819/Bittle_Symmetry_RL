'''
    Copy from rl_games runner.py
'''
from bittle_rl_gym.utils.helpers import class_to_dict, write_dict_to_yaml
from bittle_rl_gym.cfg.bittle_official_config import BittleOfficialConfig
from bittle_rl_gym import create_bittle_official_env
import os
import time
import glob
import argparse, os, yaml
from distutils.util import strtobool
from rl_games.common import env_configurations, vecenv
from rl_games.common.env_configurations import register


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

'''
    Copy from isaacgymenvs: to inherit vecenv class from rl_games
'''
class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

    def set_train_info(self, env_frames, *args_, **kwargs_):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        if hasattr(self.env, 'set_train_info'):
            self.env.set_train_info(env_frames, *args_, **kwargs_)

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        if hasattr(self.env, 'get_env_state'):
            return self.env.get_env_state()
        else:
            return None

    def set_env_state(self, env_state):
        if hasattr(self.env, 'set_env_state'):
            self.env.set_env_state(env_state)

vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))


'''
    Other helper functions
'''
# Write the configurations to the logging directory.
def save_cfgdict_to_dir(cfg_dict, target_root_dir, file_name):
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
    write_dict_to_yaml(cfg_dict, os.path.join(target_root_dir, file_name))


# Get the path of the latest trained policy.
def get_latest_policy_path(env_name, log_root = 'runs/'):
    history_exp_paths = list(sorted(glob.glob(os.path.join(log_root, f'*/*/{env_name}.pth'), recursive = True)))
    if len(history_exp_paths) > 0:
        return history_exp_paths[-1]
    else:
        return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument("--seed", type=int, default=0, required=False, 
    #                 help="random seed, if larger than 0 will overwrite the value in yaml config")
    # ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
    # ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play (test) network", action='store_true')
    ap.add_argument("-r", "--record", required=False, help="record a video for evaluation", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to the training config")
    # ap.add_argument("-na", "--num_actors", type=int, default=0, required=False,
    #                 help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
    # ap.add_argument("-s", "--sigma", type=float, required=False, help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
    # ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="if toggled, this experiment will be tracked with Weights and Biases")
    # ap.add_argument("--wandb-project-name", type=str, default="rl_games",
    #     help="the wandb's project name")
    # ap.add_argument("--wandb-entity", type=str, default=None,
    #     help="the entity (team) of wandb's project")

    args = vars(ap.parse_args())

    '''
        Register the bittle environment to rl_games.
    '''
    def create_env(**kwargs):
        env_cfg = BittleOfficialConfig()
        is_train = not args['play']
        if is_train:
            # Do not record any videos or visualize when training
            record_video = False
            headless = True
            
            log_dir = os.path.join(exp_root_path, config['params']['config']['full_experiment_name'])
            if os.path.exists(log_dir):
                save_cfgdict_to_dir(class_to_dict(env_cfg), log_dir, 'env.yaml')

        else:
            record_video = args['record']
            headless = record_video
            # Change some parameters
            env_cfg.init_state.noise.add_noise = False
            env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)

        env = create_bittle_official_env(env_cfg, headless, record_video)
        return env

    register(
        'Bittle',
        {
            'env_creator' : lambda **kwargs : create_env(**kwargs),
            'vecenv_type' : 'RLGPU'
        }
    )

    exp_root_path = "runs"
    os.makedirs(exp_root_path, exist_ok=True)

    config_name = args['file']
    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)

        # if args['num_actors'] > 0:
        #     config['params']['config']['num_actors'] = args['num_actors']

        # if args['seed'] > 0:
        #     config['params']['seed'] = args['seed']
        #     config['params']['config']['env_config']['seed'] = args['seed']

        args['train'] = not args['play']

        if args['train']:
            # Label the logging directory with the current time
            curr_time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
            config['params']['config']['full_experiment_name'] = f"{config['params']['config']['env_name']}{curr_time_str}"

            # Write the configuration files to the logging directory.
            save_cfgdict_to_dir(config, os.path.join(exp_root_path, config['params']['config']['full_experiment_name']), 'train.yaml')

        else:
            # Try to find the latest trained policy if not given.
            if args['checkpoint'] is None:
                latest_policy_path = get_latest_policy_path(config['params']['config']['env_name'], exp_root_path)
                if latest_policy_path is not None:
                    args['checkpoint'] = latest_policy_path
                    print(f'Play: Load the latest policy from {latest_policy_path}!')
                else:
                    print('Play: No checkpoint loaded for testing!')

        from rl_games.torch_runner import Runner

        try:
            import ray
        except ImportError:
            pass
        else:
            ray.init(object_store_memory=1024*1024*1000)

        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    # global_rank = int(os.getenv("RANK", "0"))
    # if args["track"] and global_rank == 0:
    #     import wandb
    #     wandb.init(
    #         project=args["wandb_project_name"],
    #         entity=args["wandb_entity"],
    #         sync_tensorboard=True,
    #         config=config,
    #         monitor_gym=True,
    #         save_code=True,
    #     )

    runner.run(args)

    try:
        import ray
    except ImportError:
        pass
    else:
        ray.shutdown()

    # if args["track"] and global_rank == 0:
    #     wandb.finish()