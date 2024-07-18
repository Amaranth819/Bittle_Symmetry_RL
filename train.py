from bittle_rl_gym.env.bittle import Bittle
from bittle_rl_gym.env.bittle_config import BittleConfig
from bittle_rl_gym.utils.helpers import class_to_dict
from isaacgym import gymapi, gymutil
import torch
import time


def parse_sim_params(cfg):
    physics_engine = gymapi.SIM_PHYSX

    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    sim_params.physx.use_gpu = torch.cuda.is_available()
    # sim_params.physx.num_subscenes = 4
    sim_params.use_gpu_pipeline = torch.cuda.is_available()

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.num_threads = 4

    return physics_engine, sim_params


def main():
    cfg = BittleConfig()
    sim_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    physics_engine, sim_params = parse_sim_params(class_to_dict(cfg))

    env = Bittle(
        cfg = cfg, 
        sim_params = sim_params, 
        physics_engine = physics_engine,
        sim_device = sim_device,
        headless = False
    )

    env.render()
    env.reset()
    print(env.num_envs)
    env._print_PD_gains()
    for i in range(1000):
        _, _, _, reset, info = env.step(1 * torch.randn((env.num_envs, env.num_actions)))

        if i == 100:
            start = time.time()
            env._update_PD_gains(env.P_gains.cpu().numpy() * 10, env.D_gains.cpu().numpy() * 10)
            print(time.time() - start)
        
        if i == 120:
            env._print_PD_gains()


if __name__ == '__main__':
    main()