from bittle_rl_gym import create_bittle_env
from rsl_rl_cfg import create_alg_runner



def train():
    env = create_bittle_env(headless = True)
    alg, cfg = create_alg_runner(env, log_root = 'exps/')
    alg.learn(cfg.runner.max_iterations, init_at_random_ep_len = True)


def test(pretrained_model_path = '', record_video = False):
    pass


if __name__ == '__main__':
    train()