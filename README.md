# Bittle_Symmetry_RL

Implement Bittle isaacgym environment based on Unitree_rl_gym: https://support.unitree.com/home/zh/developer/rl_example



### Possible bugs
1. Using Numpy >= 1.20 may report "AttributeError: module 'numpy' has no attribute 'float'" at get_axis_params() (line 135 in isaacgym/torch_utils.py). **Solution**: change "dtype=np.float" to "dtype=float".