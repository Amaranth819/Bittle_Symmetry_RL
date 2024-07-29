import gym.wrappers.record_video
from bittle_rl_gym import create_bittle_env
from rsl_rl_cfg import create_alg_runner, BittlePPO
from bittle_rl_gym.env.bittle_config import BittleConfig
from bittle_rl_gym.utils.helpers import write_dict_to_yaml, class_to_dict
import gym


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


class BittleRecordVideo(gym.wrappers.record_video.RecordVideo):
    def step(self, action):
        observations, privileged_observations, rewards, dones, infos = self.env.step(action)

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return observations, privileged_observations, rewards, dones, infos


def test(pretrained_model_path = None, record_video = True):
    env_cfg = BittleConfig()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    # env_cfg.viewer.ref_env = 0
    env = create_bittle_env(env_cfg, headless = record_video)
    if record_video:
        env._create_camera(env_idx = 0, p = [0.3, 0, 0], axis = [0, 0, 1], angle = 180.0, follow = 'FOLLOW_POSITION')
        env._wrap_cameras()

    alg_cfg = BittlePPO()
    if pretrained_model_path is not None:
        alg_cfg.runner.resume = True
        alg_cfg.runner.resume_path = pretrained_model_path
    alg = create_alg_runner(env, alg_cfg, log_root = None)
    policy = alg.get_inference_policy(device = env.device)

    obs, _ = env.reset()
    for i in range(100):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
    
    if record_video:
        env.save_record_video(name = 'video')


if __name__ == '__main__':
    # train()
    test()