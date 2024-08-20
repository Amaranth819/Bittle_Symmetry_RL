from bittle_rl_gym import create_bittle_official_env
from rsl_rl_cfg import create_alg_runner, BittlePPO
# from bittle_rl_gym.cfg.bittle_aiwintermuteai_config import BittleAIWintermuteAIConfig
from bittle_rl_gym.cfg.bittle_official_config import BittleOfficialConfig
from bittle_rl_gym.utils.helpers import write_dict_to_yaml, class_to_dict
import torch
import glob
import os
import argparse


# Helper functions
def save_cfgs_to_exp_dir(env_cfg, alg_cfg, target_root_dir):
    import os
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
    
    write_dict_to_yaml(class_to_dict(env_cfg), os.path.join(target_root_dir, 'env.yaml'))
    write_dict_to_yaml(class_to_dict(alg_cfg), os.path.join(target_root_dir, 'alg.yaml'))


def get_latest_policy_path(exp_name, log_root = 'exps/'):
    history_exp_paths = list(sorted(glob.glob(os.path.join(log_root, f'{exp_name}*'))))
    if len(history_exp_paths) > 0:
        return os.path.join(history_exp_paths[-1], 'model_final.pt')
    else:
        return None


# Train and test
def train(log_root = 'exps/'):
    env_cfg = BittleOfficialConfig()
    env = create_bittle_official_env(env_cfg, headless = True, record_video = False)
    
    alg_cfg = BittlePPO()
    alg = create_alg_runner(env, alg_cfg, log_root = log_root)

    save_cfgs_to_exp_dir(env_cfg, alg_cfg, alg.log_dir)
    alg.learn(alg_cfg.runner.max_iterations, init_at_random_ep_len = False)
    alg.save(os.path.join(alg.log_dir, 'model_final.pt'))



def test(
    pretrained_model_path = None, 
    headless = False, 
    record_video = False, 
    video_prefix = 'video'
):
    # If reporting error "[Error] [carb.gym.plugin] cudaImportExternalMemory failed on rgbImage buffer with error 999", then "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json". (https://forums.developer.nvidia.com/t/cudaimportexternalmemory-failed-on-rgbimage/212944/5)
    env_cfg = BittleOfficialConfig()
    env_cfg.init_state.noise.add_noise = False
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env = create_bittle_official_env(env_cfg, headless = headless, record_video = record_video)

    if record_video:
        env._create_camera(env_idx = 0)

    alg_cfg = BittlePPO()
    if pretrained_model_path is None:
        # Try to find the latest trained model instead.
        alg_class_name = alg_cfg.runner.algorithm_class_name
        exp_name = alg_cfg.runner.experiment_name
        pretrained_model_path = get_latest_policy_path(f'{exp_name}{alg_class_name}')
    if pretrained_model_path is not None:
        # Find a target policy to load.
        alg_cfg.runner.resume = True
        alg_cfg.runner.resume_path = pretrained_model_path

    alg = create_alg_runner(env, alg_cfg, log_root = None)
    policy = alg.get_inference_policy(device = env.device)

    # obs, _ = env.reset()
    obs = env.compute_observations()
    total_rews, total_steps = 0, 0
    for idx in range(env.max_episode_length):
        actions = policy(obs.detach()).detach()
        obs, _, rews, dones, infos = env.step(actions)
        total_rews += rews[0].item()
        total_steps += 1
        # print(env._get_contact_forces(env.foot_indices))
        print(torch.min(torch.abs(env.dof_vel), dim = -1)[0])
        if dones[0].item() == True:
            print(f'Rewards = {total_rews} | Steps = {total_steps}')
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



def test_walking_traj():
    # start from kd^2 + 4*kp*ki = 0, where ki = 1 if not using the integral part. 
    env_cfg = BittleOfficialConfig()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env = create_bittle_official_env(env_cfg, headless = True, record_video = True)
    env.reset()

    env._create_camera(env_idx = 0)

    import numpy as np
    def read_traj():
        traj_data = np.asarray(np.genfromtxt('gait_sequence.csv', delimiter = ',')).astype('float32')[:, :-1]
        traj_data = traj_data * np.pi / 180
        # index in csv: [lfs, rfs, rft, lft, lrs, rrs, rrt, lrt]
        traj_data = (traj_data[:, [0,3,4,7,1,2,5,6]] - env.default_dof_pos[..., 1:].cpu().numpy()) / env.cfg.control.action_scale
        traj_data = np.concatenate([np.zeros((traj_data.shape[0], 1)), traj_data], axis = -1)
        traj_data = torch.from_numpy(traj_data).float().to(env.device)
        return traj_data

    trajs = read_traj().unsqueeze(1)

    for i in range(trajs.size(0)):
        env.step(trajs[i])
        print(trajs[i])

    env.save_record_video(name = 'traj', postfix = 'mp4')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--record_video', action = 'store_true')

    args = parser.parse_args()
    if args.test:
        test(pretrained_model_path = None, headless = args.record_video, record_video = args.record_video, video_prefix = 'video')
    else:
        train(log_root = 'exps/')


if __name__ == '__main__':
    main()
    # test_walking_traj()