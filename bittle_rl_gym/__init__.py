# from bittle_rl_gym.env.bittle_aiwintermuteai import BittleAIWintermuteAI
# from bittle_rl_gym.cfg.bittle_aiwintermuteai_config import BittleAIWintermuteAIConfig
from bittle_rl_gym.env.bittle_official import BittleOfficial
from bittle_rl_gym.cfg.bittle_official_config import BittleOfficialConfig
from bittle_rl_gym.utils.helpers import class_to_dict
from isaacgym import gymapi, gymutil
import torch


# def create_bittle_aiwintermuteai_env(cfg = BittleAIWintermuteAIConfig(), headless = True):
#     sim_device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

#     # Simulation parameters
#     sim_params = gymapi.SimParams()
#     sim_params.physx.use_gpu = torch.cuda.is_available()
#     sim_params.use_gpu_pipeline = torch.cuda.is_available()

#     cfg_dict = class_to_dict(cfg)
#     if 'sim' in cfg_dict:
#         gymutil.parse_sim_config(cfg_dict['sim'], sim_params)

#     physics_engine = gymapi.SIM_PHYSX # gymapi.SIM_FLEX
#     if physics_engine == gymapi.SIM_PHYSX:
#         sim_params.physx.num_threads = cfg.sim.physx.num_threads
#         sim_params.physx.num_subscenes = cfg.sim.physx.num_subscenes

#     # Create the environment
#     env = BittleAIWintermuteAI(
#         cfg = cfg, 
#         sim_params = sim_params, 
#         physics_engine = physics_engine,
#         sim_device = sim_device,
#         headless = headless, # No visualization if true,
#     )

#     return env



def create_bittle_official_env(cfg = BittleOfficialConfig(), headless = True, record_video = False):
    sim_device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

    # Simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.physx.use_gpu = torch.cuda.is_available()
    sim_params.use_gpu_pipeline = torch.cuda.is_available()

    cfg_dict = class_to_dict(cfg)
    if 'sim' in cfg_dict:
        gymutil.parse_sim_config(cfg_dict['sim'], sim_params)

    physics_engine = gymapi.SIM_PHYSX # gymapi.SIM_FLEX
    if physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.num_threads = cfg.sim.physx.num_threads
        sim_params.physx.num_subscenes = cfg.sim.physx.num_subscenes

    # Create the environment
    env = BittleOfficial(
        cfg = cfg, 
        sim_params = sim_params, 
        physics_engine = physics_engine,
        sim_device = sim_device,
        headless = headless, # No visualization if true,
        record_video = record_video
    )

    return env