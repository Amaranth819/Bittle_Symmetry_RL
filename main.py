import gym.wrappers.record_video
from bittle_rl_gym import create_bittle_official_env
from rsl_rl_cfg import create_alg_runner, BittlePPO
# from bittle_rl_gym.env.bittle_config import BittleConfig
from bittle_rl_gym.env.bittle_official_config import BittleOfficialConfig
from bittle_rl_gym.utils.helpers import write_dict_to_yaml, class_to_dict
import gym
import torch


def save_cfgs_to_exp_dir(env_cfg, alg_cfg, target_root_dir):
    import os
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
    
    write_dict_to_yaml(class_to_dict(env_cfg), os.path.join(target_root_dir, 'env.yaml'))
    write_dict_to_yaml(class_to_dict(alg_cfg), os.path.join(target_root_dir, 'alg.yaml'))


def train(log_root = 'exps/'):
    env_cfg = BittleOfficialConfig()
    env = create_bittle_official_env(env_cfg, headless = True)
    
    alg_cfg = BittlePPO()
    alg = create_alg_runner(env, alg_cfg, log_root = log_root)

    save_cfgs_to_exp_dir(env_cfg, alg_cfg, alg.log_dir)
    alg.learn(alg_cfg.runner.max_iterations, init_at_random_ep_len = True)



def test(pretrained_model_path = None, record_video = True, video_prefix = 'video'):
    # If reporting error "[Error] [carb.gym.plugin] cudaImportExternalMemory failed on rgbImage buffer with error 999", then "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json". (https://forums.developer.nvidia.com/t/cudaimportexternalmemory-failed-on-rgbimage/212944/5)
    env_cfg = BittleOfficialConfig()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    # env_cfg.viewer.ref_env = 0
    env = create_bittle_official_env(env_cfg, headless = record_video)
    if record_video:
        env._create_camera(env_idx = 0)
        env._wrap_cameras()

    alg_cfg = BittlePPO()
    if pretrained_model_path is not None:
        alg_cfg.runner.resume = True
        alg_cfg.runner.resume_path = pretrained_model_path
    alg = create_alg_runner(env, alg_cfg, log_root = None)
    policy = alg.get_inference_policy(device = env.device)

    obs, _ = env.reset()
    for _ in range(env.max_episode_length // 10):
        actions = policy(obs.detach())
        # actions = torch.zeros_like(env.actions)
        obs, _, rews, dones, infos = env.step(actions.detach())
        if dones[0].item():
            break
    
    if record_video:
        env.save_record_video(name = video_prefix)


if __name__ == '__main__':
    # train()
    # test('exps/BittlePPO-2024-07-29-20:15:13/model_100.pt', video_prefix = 'video')
    test()