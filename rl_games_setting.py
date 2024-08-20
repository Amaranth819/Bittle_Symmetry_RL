import yaml
import bittle_rl_gym # Avoid ImportError: PyTorch was imported before isaacgym modules. Please import torch after isaacgym modules.
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv

from bittle_rl_gym.utils.helpers import read_dict_from_yaml
from bittle_rl_gym import create_bittle_official_env


env_configurations.register(
    'rlgpu', 
    {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_bittle_official_env(**kwargs),
    }
)

bittle_train_cfg = read_dict_from_yaml('rl_games_bittle.yaml')
runner = Runner()
runner.load(bittle_train_cfg)
runner.run({
    'train': True,
})

# config = walker_config
# config['params']['config']['full_experiment_name'] = 'Walker2d_mujoco'
# config['params']['config']['max_epochs'] = 500
# config['params']['config']['horizon_length'] = 512
# config['params']['config']['num_actors'] = 8
# config['params']['config']['minibatch_size'] = 1024
# runner = Runner()
# runner.load(config)
# runner.run({
#     'train': True,
# })