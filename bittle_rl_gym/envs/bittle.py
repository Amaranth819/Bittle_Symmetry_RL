from bittle_rl_gym.envs.base_task import BaseTask



class Bittle(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


        if not self.headless:
            # Set camera
            pass

    

    def _init_buffers(self):
        self.obs_buf = 