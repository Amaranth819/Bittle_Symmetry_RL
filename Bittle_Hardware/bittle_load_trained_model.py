from bittle_rl_gym.utils.helpers import read_dict_from_yaml
from bittle_rl_gym.cfg.bittle_official_config import BittleOfficialConfig
from rl_games.torch_runner import _restore
from rl_games.torch_runner import Runner
from Bittle_Hardware.bittle_rl_games_train import register_bittle_env
import os


def read_trained_policy_from_rootdir(checkpoint_path, train_cfg_yaml_path):
    args = {}
    args['checkpoint'] = checkpoint_path
    args['play'] = True
    args['record'] = False
    args['file'] = train_cfg_yaml_path

    register_bittle_env(args)
    runner = Runner()
    runner.load(read_dict_from_yaml(args['file']))

    player = runner.create_player()
    _restore(player, args)
    return player


if __name__ == '__main__':
    player = read_trained_policy_from_rootdir(
        checkpoint_path = 'runs/Bittle2024-09-02-18:03:58/nn/Bittle.pth',
        train_cfg_yaml_path = 'runs/Bittle2024-09-02-18:03:58/train.yaml'
    )
    print(player.model)

    import torch
    import time
    obs = torch.randn(1, 45).cuda()

    player.init_rnn()

    # 0.088s
    start = time.time()
    act = player.get_action(obs, True)
    end = time.time()
    print(act.size())
    print(end - start)

    # # # 0.083s
    # rnn_states = player.states
    # model = player.model
    # input_dict = {
    #     'is_train': False,
    #     'prev_actions': None, 
    #     'obs' : x,
    #     'rnn_states' : rnn_states
    # }
    # start = time.time()
    # res_dict = model(input_dict)
    # end = time.time()
    # rnn_states = res_dict['rnn_states']
    # print(res_dict['mus'].size(), res_dict['actions'].size())
    # print(end - start)