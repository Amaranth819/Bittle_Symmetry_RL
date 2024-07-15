from bittle_rl_gym.env.bittle import Bittle
from bittle_rl_gym.env.bittle_config import BittleConfig
from bittle_rl_gym.utils.helpers import class_to_dict
from isaacgym import gymapi
import torch


def main():
    cfg = BittleConfig()
    sim_params = gymapi.SimParams()
    physics_engine = gymapi.SIM_PHYSX
    sim_device = 'cuda:0'

    env = Bittle(
        cfg = cfg, 
        sim_params = sim_params, 
        physics_engine = physics_engine,
        sim_device = sim_device,
        headless = True
    )

    env.render()

    for i in range(1000):
        _, _, done, info = env.step(torch.zeros(cfg.env.num_envs, cfg.env.num_actions))
        print(i, done)


class Test(object):
    def __init__(self) -> None:
        self.x = 1
        self.y = 2


if __name__ == '__main__':
    t = Test()
    print(t.x)