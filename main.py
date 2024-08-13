from bittle_rl_gym import create_bittle_official_env
from rsl_rl_cfg import create_alg_runner, BittlePPO
# from bittle_rl_gym.cfg.bittle_aiwintermuteai_config import BittleAIWintermuteAIConfig
from bittle_rl_gym.cfg.bittle_official_config import BittleOfficialConfig
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
    env = create_bittle_official_env(env_cfg, headless = True, record_video = False)
    
    alg_cfg = BittlePPO()
    alg = create_alg_runner(env, alg_cfg, log_root = log_root)

    save_cfgs_to_exp_dir(env_cfg, alg_cfg, alg.log_dir)
    alg.learn(alg_cfg.runner.max_iterations, init_at_random_ep_len = False)



def test(pretrained_model_path = None, headless = False, record_video = True, video_prefix = 'video'):
    # If reporting error "[Error] [carb.gym.plugin] cudaImportExternalMemory failed on rgbImage buffer with error 999", then "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json". (https://forums.developer.nvidia.com/t/cudaimportexternalmemory-failed-on-rgbimage/212944/5)
    env_cfg = BittleOfficialConfig()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    # env_cfg.viewer.ref_env = 0
    env = create_bittle_official_env(env_cfg, headless = headless, record_video = record_video)
    if record_video:
        env._create_camera(env_idx = 0)

    alg_cfg = BittlePPO()
    if pretrained_model_path is not None:
        alg_cfg.runner.resume = True
        alg_cfg.runner.resume_path = pretrained_model_path
    alg = create_alg_runner(env, alg_cfg, log_root = None)
    policy = alg.get_inference_policy(device = env.device)

    jp_errors = []

    obs, _ = env.reset()
    for idx in range(env.max_episode_length):
        # actions = policy(obs.detach()).detach()
        actions = torch.randn(env.num_envs, env.num_actions).clamp(-1, 1).to(env.device)
        # print(actions.min(), actions.max())

        # Tune pd gains
        target_jp = actions * env.cfg.control.action_scale + env.default_dof_pos

        obs, _, rews, dones, infos = env.step(actions)

        # 
        real_jp = env.dof_pos

        jp_errors.append((real_jp - target_jp)[0].cpu().numpy())


        # print(env.torques.min(), env.torques.max())
        if dones[0].item() == True:
            print(idx, dones)
            break
    
    if record_video:
        env.save_record_video(name = video_prefix)

    
    import numpy as np
    import matplotlib.pyplot as plt
    jp_errors = np.stack(jp_errors, axis = -1) # [num_dofs, num_steps]
    num_dofs, num_steps = jp_errors.shape
    fig, ax = plt.subplots(num_dofs)
    fig.set_figheight(num_dofs * 2)
    fig.set_figwidth(8)
    for idx in range(num_dofs):
        ax[idx].plot(np.arange(num_steps), jp_errors[idx])
        ax[idx].set_title(env.dof_names[idx])
    plt.tight_layout()
    plt.savefig('pd.png')
    plt.close()


if __name__ == '__main__':
    # train()
    # test('exps/BittlePPO-2024-08-12-21:55:27/model_500.pt', headless = True, record_video = True, video_prefix = 'video')
    test(headless = True, record_video = True)