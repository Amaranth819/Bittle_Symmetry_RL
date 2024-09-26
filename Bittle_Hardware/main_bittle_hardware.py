import sys
sys.path.append("..")


# In terminal, give access to the serial port before you run the code.
# E.g., sudo chmod 777 /dev/ttyACM0


from Bittle_Hardware.bittle_load_trained_model import read_trained_policy_from_rootdir
import time
import logging
import numpy as np
import torch

from bittle_ardSerial import *
from bittle_esekf import *
from bittle_send_require import *
import threading
# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Step 1: Initialize Port Connections
goodPorts = {}
connectPort(goodPorts)
t = threading.Thread(target=keepCheckingPort, args=(goodPorts,))
t.start()
print('GoodPorts:', goodPorts)

# Step 2: Set the Robot to Initial Joint Position
initial_target_position = np.array([-0.6981,            # head
                                    -0.7854,   1.571,   # lf_s, lf_t
                                     0.7854,   1.571,   # lr_s, lr_t
                                    -0.7854,   1.571,   # rf_s, rf_t
                                     0.7854,   1.571])  # rr_s, rr_t
send_joint_command(goodPorts, initial_target_position)
time.sleep(3)

# imu_linear_acc, imu_angular_vel, dof_positions = require_gyro_and_dof_data(goodPorts)


# Step 3: Initialize IMU Parameters and EKF
imu_params = ImuParameters()
imu_params.sigma_a_n = 0.00981   # Accelerometer noise (m/s²/√Hz)
imu_params.sigma_w_n = 0.00087   # Gyroscope noise (rad/s/√Hz)
imu_params.sigma_a_b = 0.000981  # Accelerometer bias instability (m/s²)
imu_params.sigma_w_b = 0.000523  # Gyroscope bias instability (rad/s)

init_state = np.zeros(19)  # Initialize nominal state: [p, q, v, a_b, w_b, g]
init_state[6] = 9.81  # Set gravity in the z-axis
init_state[10:13] = np.array([imu_params.sigma_a_b] * 3)  # Accelerometer bias
init_state[13:16] = np.array([imu_params.sigma_w_b] * 3)  # Gyroscope bias

# Initialize ESEKF with nominal state and IMU parameters
esekf = ESEKF(init_nominal_state=init_state, imu_parameters=imu_params)

# # # Step 4: Load Neural Network Model
# load_run_path = '/home/dlar58/Documents/IsaacGym_projects/Bittle_Symmetry_RL/good_runs/Bounding_Bittle2024-09-11-17:13:20'
# checkpoint_path = os.path.join(load_run_path, 'nn/last_Bittle_ep_20000_rew__607.70996_.pth')
# train_cfg_yaml_path = os.path.join(load_run_path, 'train.yaml')
# player = read_trained_policy_from_rootdir(checkpoint_path, train_cfg_yaml_path)
# print(player.model)
# player.init_rnn()


# obs = torch.randn(1, 45).cuda()
# start = time.time()
# act = player.get_action(obs, True)
# end = time.time()
# print(act.size())
# print(end - start)


command_linvel = np.array([0.15, 0.0]) # x, y
command_angvel = np.array([0.0]) # yaw
prev_action = np.zeros(9)
gait_period = 0.2576 * np.exp(-0.9829 * np.abs(command_linvel[0])) # * (1 + random_scale * abs_cmd_forward_linvel * 0.25)
duty_factor = 0.5588 * np.exp(-0.6875 * np.abs(command_linvel[0])) # * (1 + random_scale * abs_cmd_forward_linvel * 0.25)
init_foot_thetas = np.array([0, 0.5, 0, 0.5]) # lf, lr, rf, rr
phase_ratios = np.array([1 - duty_factor, duty_factor])
dof_vel = np.zeros(9)


def _get_foot_phis(phi, cmd_linvel):
    # Copied from BittleOfficial
    sign = 1 if cmd_linvel >= 0 else -1
    return np.abs(((phi + init_foot_thetas) * sign) % sign)

import pickle
with open('bounding0.15-fp.pkl', 'rb') as handle:
    data = pickle.load(handle)['actions']


# Initialize previous DOF positions and time
dof_pos_last = None
last_iteration_time = time.time()
start_program_time = time.time()
for i in range(30):  # Example of 100 iterations
    # Step 1 (inside loop): Start iteration and time tracking
    iteration_start_time = time.time()
    logging.info(f"Iteration {i+1} start")

    # Step 2 (inside loop): Request and decode serial data (Gyro, Linear Acc, and DOF Positions)
    imu_linear_acc, imu_angular_vel, dof_positions = require_gyro_and_dof_data(goodPorts)
    
    if imu_linear_acc is None or imu_angular_vel is None or dof_positions is None:
        logging.warning("Invalid serial data, skipping iteration")
        continue

    # Step 3 (inside loop): ESEKF state prediction
    imu_measurement = np.hstack(([0], imu_angular_vel, imu_linear_acc))
    esekf.predict(imu_measurement)  # ESEKF prediction

    # Step 4 (inside loop): Access the updated nominal state
    predicted_state = esekf.nominal_state

    # Step 5 (inside loop): Extract projected gravity, linear velocity, and angular velocity
    projected_gravity, linear_velocity, angular_velocity = esekf.extract_state_information(predicted_state, imu_angular_vel)

    # Prepare Observation for the NN:           # base_lin_vels,
                                                # base_ang_vels,
                                                # dof_pos,
                                                # dof_vel,
                                                # base_proj_grav,
                                                # cmd_lin_vels,
                                                # cmd_ang_vels,
                                                # prev_actions,
                                                # foot_phis_sin,    
                                                # phase_ratios

    # Get the current time of program execution
    curr_program_time = time.time()
    phi = (curr_program_time - start_program_time) / gait_period
    foot_phis = _get_foot_phis(phi, command_linvel[0])
    foot_phis_sin = np.sin(2 * np.pi * foot_phis)
    
    # # Step 5 (inside loop): Observation-Action Mapping Using Neural-Network
    # obs = np.concatenate([linear_velocity, angular_velocity, dof_positions, dof_vel, projected_gravity, command_linvel, command_angvel, prev_action, foot_phis_sin, phase_ratios])
    # curr_action = player.get_action(torch.from_numpy(obs).float().unsqueeze(0).cuda(), True).detach().cpu().squeeze(0).numpy()
    # prev_action = np.copy(curr_action)
    # print(f"curr_action {curr_action}")
    # # # load from neural network mapping
    # target_jpos_urdf_raw = np.clip(curr_action, -1, 1) * 0.5 + initial_target_position

    # bounding: 30, load from pre-generated data
    target_jpos_urdf_raw = np.clip(data[..., i*30], -1, 1) * 0.5 + initial_target_position



    print(f"target_jpos_urdf_raw {target_jpos_urdf_raw}")


    # # Step 8 (inside loop): Send joint commands to Bittle
    send_joint_command(goodPorts, target_jpos_urdf_raw)
    # time.sleep(0.00)

    # Step 9 (inside loop): Calculate DOF velocity
    current_time = time.time()
    dt = current_time - last_iteration_time
    
    if dof_pos_last is not None and dt > 0:
        dof_vel = (dof_positions - dof_pos_last) / dt
        logging.info(f"DOF Velocity: {dof_vel}")

    # Update the last positions and time for the next iteration
    dof_pos_last = dof_positions
    last_iteration_time = current_time

    # Step 10 (inside loop): Calculate total iteration time
    iteration_time = time.time() - iteration_start_time
    logging.info(f"Iteration {i+1}: Total Iteration Time (dt): {iteration_time:.4f} seconds")

    # Step 11 (inside loop): Sleep to simulate real-time data
    time.sleep(0.0)  # Simulate a 100ms delay



closeAllSerial(goodPorts)
logger.info("finish!")
os._exit(0)
