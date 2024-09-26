# Bittle_Symmetry_RL

### About this repository

This repository is the implementation of paper "Symmetry-Guided Reinforcement Learning for Versatile Quadrupedal Gait Generation". The code is based on [unitree_rl_gym](https://support.unitree.com/home/zh/developer/rl_example), [Isaac Gym](https://developer.nvidia.com/isaac-gym) and [Isaac Gym Benchmark Environments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

**Authors**: Xulin Chen (<xchen168@syr.edu>), Jiayu Ding (<jding14@syr.edu>)

**Instructors**: Zhenyu Gan, Garrett E. Katz


This project was initially developed by [Dynamic Locomotion and Robotics Lab (DLAR Lab)](https://dlarlab.syr.edu/) at Syracuse University.

_____

### Features
- **Symmetry-based Reward Design**: Incorporate three symmetries (temporal time-reversal and morphological symmetry) into the reward function design, and train 4 gaits (trotting, bounding, half-bounding and galloping) for a quadrupedal robot [Bittle](https://www.petoi.com/pages/bittle-smart-robot-dog-model-overview).

_____


### Publication
This work has been submitted to ICRA 2025. If you use this work in an academic context, please cite the following publication: 

_____



### Organization of project files
- `rl_games_train.py`: The script of policy training / testing.
- `rl_games_bittle.yaml`: The hyperparameters for the policy optimization algorithm (PPO + LSTM by default). Need rl_games to be installed.
- `bittle_rl_gym/`: The directory of the simulated Bittle environment.
- `bittle_rl_gym/env/bittle_official.py`: The Bittle environment class.
- `bittle_rl_gym/cfg/bittle_official_config.py`: The hyperparameters for Bittle environment class, mainly including the physical parameters, gait parameters and reward design.
- `Bittle_Hardware/`: The directory of the hardware test module.

_____



### Usage

**Prerequisite**
1. Download Isaac Gym from [here](https://developer.nvidia.com/isaac-gym) and follow the installation instructions. Recommend using an individual conda environment. 
2. Once Isaac Gym is properly installed, download this repository and install the required packages by `pip install -r requirements.txt`.

**Train / test policy on simulation**
1. Run `python rl_games_train.py -f rl_games_bittle.yaml` to train a control policy. The trained policies are saved under `runs/BittleYYYY-MM-DD-HH:MM:SS`, where `nn/Bittle.pth` saves the policy achieving the best rewards, `nn/last_Bittle_ep_x_rew_y.pth` files are the policies at epoch $x$ achieving reward $y$, and `summaries/` includes the training log files that can be visualized by tensorboard. 
2. Run `python rl_games_train.py -p -r -f rl_games_bittle.yaml` to record a video for the latest trained policy. To visualize a specific policy, run `python rl_games_train.py -p -r -f rl_games_bittle.yaml -c path/to/model`. To visualize with Issac Gym GUI without recording, ignore the argument `-r`.

**Hardware test (currently only support open-loop control)**
1. Follow [Petoi Doc Center](https://docs.petoi.com/bluetooth-connection) to connect the computer to Bittle via bluetooth.
2. Go to the directory by `cd Bittle_Hardware/` and run `python main_bittle_hardware.py`.

_____



### Known bugs
1. Using Numpy >= 1.20 may report "AttributeError: module 'numpy' has no attribute 'float'" at get_axis_params() (line 135 in isaacgym/torch_utils.py). **Solution**: change "dtype=np.float" to "dtype=float" at line 135 in isaacgym/torch_utils.py.

_____

### Summary of work
<!-- <video src='Sym_Guided_RL_Video.mp4' width=640/> -->