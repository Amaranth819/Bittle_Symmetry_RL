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





import pickle
with open('bounding0.15-fp.pkl', 'rb') as handle:
    data = pickle.load(handle)['actions']


for i in range(30):  # Example of 100 iterations
    # Step 1 (inside loop): Start iteration and time tracking
    iteration_start_time = time.time()
    logging.info(f"Iteration {i+1} start")

    # Step 2 (inside loop): Request and decode serial data (Gyro, Linear Acc, and DOF Positions)
    imu_linear_acc, imu_angular_vel, dof_positions = require_gyro_and_dof_data(goodPorts)
    
    if imu_linear_acc is None or imu_angular_vel is None or dof_positions is None:
        logging.warning("Invalid serial data, skipping iteration")
        continue

   
    # # bounding: 30, load from pre-generated data
    target_jpos_urdf_raw = np.clip(data[..., i*30], -1, 1) * 0.5 + initial_target_position


    print(f"target_jpos_urdf_raw {target_jpos_urdf_raw}")


    # # Step 8 (inside loop): Send joint commands to Bittle
    send_joint_command(goodPorts, target_jpos_urdf_raw)
    # time.sleep(0.00)


    # Step 11 (inside loop): Sleep to simulate real-time data
    time.sleep(0.0)  # Simulate a 100ms delay



closeAllSerial(goodPorts)
logger.info("finish!")
os._exit(0)
