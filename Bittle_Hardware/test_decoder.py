import sys
sys.path.append("..")

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

import numpy as np

import numpy as np

# Define initial and final target positions in radians
initial_target_position = np.array([-0.6981,   # head
                                    -0.7854,  1.571,  # lf_s, lf_t
                                     0.7854,  1.571,  # lr_s, lr_t
                                    -0.7854,  1.571,  # rf_s, rf_t
                                     0.7854,  1.571]) # rr_s, rr_t

final_target_position = np.array([0.1,       # head
                                  -0,  1.7,  # lf_s, lf_t
                                   0,  0.8,  # lr_s, lr_t
                                  -0,  0.3,  # rf_s, rf_t
                                   0,  1.5]) # rr_s, rr_t

# Number of intermediate steps (10)
num_steps = 100

# Create a 2D array with rows interpolating between the initial and final positions
interpolated_positions = np.linspace(initial_target_position, final_target_position, num=num_steps+2)

# Convert the positions from radians to degrees
# interpolated_positions_deg = np.degrees(interpolated_positions)

goodPorts = {}
connectPort(goodPorts)
t = threading.Thread(target=keepCheckingPort, args=(goodPorts,))
t.start()
print('GoodPorts:', goodPorts)


# Print each row in degrees
for i, row in enumerate(interpolated_positions):
    print()
    # print(f"Row {i+1}: {row}")
    # print(f"Decoded Row {i+1}: {decode_jpos_i_command(row)}")

    send_joint_command(goodPorts, row)
    imu_linear_acc, imu_angular_vel, dof_positions = require_gyro_and_dof_data(goodPorts)
    # print(f"Joint Position {i+1}: {dof_positions}")
    print()
    print(f"Joint Position {i+1} Target Error", row-dof_positions)

time.sleep(5)


closeAllSerial(goodPorts)
logger.info("finish!")
os._exit(0)


