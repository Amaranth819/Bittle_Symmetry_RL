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
    env = create_bittle_official_env(env_cfg, headless = headless, record_video = record_video)

    if record_video:
        env._create_camera(env_idx = 0)

    alg_cfg = BittlePPO()
    if pretrained_model_path is not None:
        alg_cfg.runner.resume = True
        alg_cfg.runner.resume_path = pretrained_model_path
    alg = create_alg_runner(env, alg_cfg, log_root = None)
    policy = alg.get_inference_policy(device = env.device)

    obs, _ = env.reset()
    for idx in range(env.max_episode_length):
        actions = policy(obs.detach()).detach()
        obs, _, rews, dones, infos = env.step(actions)
        # print(env.torques.min(), env.torques.max())
        if dones[0].item() == True:
            print(idx, dones)
            break
    
    if record_video:
        env.save_record_video(name = video_prefix)



def tune_pd_gains(headless = True, record_video = True, video_prefix = 'video'):
    # start from kd^2 + 4*kp*ki = 0, where ki = 1 if not using the integral part. 
    env_cfg = BittleOfficialConfig()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env = create_bittle_official_env(env_cfg, headless = headless, record_video = record_video)
    env.reset()

    if record_video:
        env._create_camera(env_idx = 0)

    # Generate a continuous trajectory
    traj_len = env.max_episode_length
    period_len = 250
    ts = torch.arange(traj_len).to(env.device) / period_len
    trajs = []
    for idx in range(env.num_dof):
        trajs.append(torch.sin(2 * torch.pi * (ts + torch.rand(size = (1,)).to(ts.device))))
    trajs = torch.stack(trajs, dim = 0).unsqueeze(0)

    target_jps = []
    real_jps = []
    for i in range(trajs.size(-1)):
        action = trajs[..., i]
        target_jps.append((action * env.cfg.control.action_scale + env.default_dof_pos)[0])
        env.step(action)
        real_jps.append(env.dof_pos[0].clone())
        # print(env._get_base_projected_gravity(env.root_states, env.gravity_vec))

    if record_video:
        env.save_record_video(video_prefix)

    # Plot
    target_jps = torch.stack(target_jps, dim = 0).cpu().numpy() # [num_steps, num_dof]
    real_jps = torch.stack(real_jps, dim = 0).cpu().numpy()
    import numpy as np
    import matplotlib.pyplot as plt
    num_steps, num_dofs = target_jps.shape
    fig, ax = plt.subplots(num_dofs)
    fig.set_figheight(num_dofs * 2)
    fig.set_figwidth(8)
    for idx in range(num_dofs):
        ax[idx].plot(np.arange(num_steps), target_jps[..., idx], label = 'target')
        ax[idx].plot(np.arange(num_steps), real_jps[..., idx], label = 'real')
        ax[idx].plot(np.arange(num_steps), target_jps[..., idx] - real_jps[..., idx], label = 'error')
        ax[idx].set_title(f'{env.dof_names[idx]}, joint {idx}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pd.png')
    plt.close()


if __name__ == '__main__':
    train()
    # test('exps/BittlePPO-2024-08-15-16:51:35/model_1500.pt', headless = True, record_video = True, video_prefix = 'video')
    # test(headless = True, record_video = True, video_prefix = 'video')
    # tune_pd_gains()