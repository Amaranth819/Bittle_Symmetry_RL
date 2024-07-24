from bittle_rl_gym import create_bittle_env
from rsl_rl_cfg import create_alg_runner, BittlePPO
from bittle_rl_gym.env.bittle_config import BittleConfig
from bittle_rl_gym.utils.helpers import write_dict_to_yaml, class_to_dict


def save_cfgs_to_exp_dir(env_cfg, alg_cfg, target_root_dir):
    import os
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
    
    write_dict_to_yaml(class_to_dict(env_cfg), os.path.join(target_root_dir, 'env.yaml'))
    write_dict_to_yaml(class_to_dict(alg_cfg), os.path.join(target_root_dir, 'alg.yaml'))


def train(log_root = 'exps/'):
    env_cfg = BittleConfig()
    env = create_bittle_env(env_cfg, headless = True)
    
    alg_cfg = BittlePPO()
    alg = create_alg_runner(env, alg_cfg, log_root = log_root)

    save_cfgs_to_exp_dir(env_cfg, alg_cfg, alg.log_dir)
    alg.learn(alg_cfg.runner.max_iterations, init_at_random_ep_len = True)


def test(pretrained_model_path, record_video = False):
    env_cfg = BittleConfig()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env = create_bittle_env(headless = False)

    alg_cfg = BittlePPO()
    alg_cfg.runner.resume = True
    alg_cfg.runner.resume_path = pretrained_model_path
    alg = create_alg_runner(env, alg_cfg, log_root = None)
    policy = alg.get_inference_policy(device = env.device)

    obs, _ = env.reset()
    for i in range(env.max_episode_length):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    train()