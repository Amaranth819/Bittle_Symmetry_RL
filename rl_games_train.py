'''
    Copy from rl_games runner.py
'''

from distutils.util import strtobool
import argparse, os, yaml
from bittle_rl_gym import create_bittle_official_env
from rl_games.common import env_configurations, vecenv

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
    Register the bittle environment to rl_games.
'''
from rl_games.common.env_configurations import register
register(
    'Bittle',
    {
        'env_creator' : lambda **kwargs : create_bittle_official_env(is_train = False, headless = True, record_video = True),
        'vecenv_type' : 'RLGPU'
    }
)


'''
    Other helper functions
'''
from bittle_rl_gym.utils.helpers import class_to_dict, write_dict_to_yaml
import os
import time
def save_cfgs_to_exp_dir(env_cfg, alg_cfg, target_root_dir):
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
    
    write_dict_to_yaml(class_to_dict(env_cfg), os.path.join(target_root_dir, 'env.yaml'))
    write_dict_to_yaml(class_to_dict(alg_cfg), os.path.join(target_root_dir, 'alg.yaml'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, required=False, 
                    help="random seed, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to config")
    ap.add_argument("-na", "--num_actors", type=int, default=0, required=False,
                    help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
    ap.add_argument("-s", "--sigma", type=float, required=False, help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
    ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    ap.add_argument("--wandb-project-name", type=str, default="rl_games",
        help="the wandb's project name")
    ap.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    args = vars(ap.parse_args())
    os.makedirs("runs", exist_ok=True)

    config_name = args['file']
    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)

        if args['num_actors'] > 0:
            config['params']['config']['num_actors'] = args['num_actors']

        if args['seed'] > 0:
            config['params']['seed'] = args['seed']
            config['params']['config']['env_config']['seed'] = args['seed']

        # Change directory name
        curr_time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        config['params']['config']['full_experiment_name'] = f"{config['params']['config']['full_experiment_name']}{curr_time_str}"

        # Change some parameters for testing
        if args['play']:
            config['params']['config']['player']['games_num']

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

    global_rank = int(os.getenv("RANK", "0"))
    if args["track"] and global_rank == 0:
        import wandb
        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )

    runner.run(args)

    try:
        import ray
    except ImportError:
        pass
    else:
        ray.shutdown()

    if args["track"] and global_rank == 0:
        wandb.finish()