
import re
from bittle_ardSerial import *
# the following skill arrays are identical to those in InstinctBittle.h

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import math

# Decoding Functions

def decode_jpos_i_command(target_jpos_urdf_raw):
    """
    Decodes the target joint positions (URDF format) into the format needed for the task.
    The joint order is: [0, 8, 12, 11, 15, 9, 13, 10, 14] (head, legs, etc.).
    
    Args:
    - target_jpos_urdf_raw: A NumPy array of target joint positions in URDF format (radians).
    
    Returns:
    - task_i: A list of joint positions in the required command format.
    """
    RAD_TO_DEG = 180 / math.pi
    target_jpos_urdf = target_jpos_urdf_raw * RAD_TO_DEG

    task_i = [
        0,  -(target_jpos_urdf[0] + 40),
        8,    target_jpos_urdf[1] + 75,
        12,  target_jpos_urdf[2] - 70,
        11, -(target_jpos_urdf[3] - 75),
        15,  target_jpos_urdf[4] - 70,
        9,    target_jpos_urdf[5] + 75,
        13,  target_jpos_urdf[6] - 70,
        10, -(target_jpos_urdf[7] - 75),
        14,  target_jpos_urdf[8] - 70
    ]
    return task_i


def decode_jpos_result(res):
    """
    Decode the joint position data from the serial response.

    Args:
    - res: The raw response from the serial port.

    Returns:
    - dof_positions_urdf: A NumPy array of joint positions in URDF format (radians).
    """
    def str_to_int(L):
        return list(map(lambda x: int(x), L))

    pattern = re.compile(r'\t|\r|\n|,|=')
    info_str = res[-1]
    info_str_list = list(filter(lambda x: x != '', pattern.split(info_str)))

    joint_indices = str_to_int(info_str_list[0:16])
    joint_positions = str_to_int(info_str_list[16:])
    joint_dict = {idx: pos for idx, pos in zip(joint_indices, joint_positions)}

    joint_order = [0, 8, 12, 11, 15, 9, 13, 10, 14]
    dof_positions_i = [joint_dict[key] for key in joint_order]

    # Convert to URDF format
    dof_positions_urdf_raw = [-dof_positions_i[0] - 40,
                              dof_positions_i[1] - 75, dof_positions_i[2] + 70,
                              -dof_positions_i[3] + 75, dof_positions_i[4] + 70,
                              dof_positions_i[5] - 75, dof_positions_i[6] + 70,
                              -dof_positions_i[7] + 75, dof_positions_i[8] + 70]

    DEG_TO_RAD = math.pi / 180  # Conversion from degrees to radians
    dof_positions_urdf = np.array(dof_positions_urdf_raw) * DEG_TO_RAD

    return dof_positions_urdf


def decode_gyro_result(res):
    """
    Decode the gyro and acceleration data from the serial response.

    Args:
    - res: The raw response from the serial port.

    Returns:
    - lin_acc: A NumPy array representing linear acceleration in m/s².
    - ang_vel: A NumPy array representing angular velocity in radians/s.
    """
    ACCEL_SENSITIVITY = 16384.0  # ±2g -> 16384 LSB/g
    GYRO_SENSITIVITY = 131.0     # ±250°/s -> 131 LSB/°/s
    G_TO_MS2 = 9.81              # Conversion from g to m/s²
    DEG_TO_RAD = math.pi / 180    # Conversion from degrees to radians

    def str_to_int(L):
        return list(map(lambda x: int(x), L))

    pattern = re.compile(r'\t|\r|\n|,|=')
    info_str = res[0]
    info_str_list = list(filter(lambda x: x != '', pattern.split(info_str)))

    lin_acc_raw = np.array(str_to_int(info_str_list[0:3]))
    ang_vel_raw = np.array(str_to_int(info_str_list[3:6]))

    lin_acc = lin_acc_raw / ACCEL_SENSITIVITY * G_TO_MS2
    ang_vel = ang_vel_raw / GYRO_SENSITIVITY * DEG_TO_RAD

    return lin_acc, ang_vel


# Send/Require Functions

def send_joint_command(goodPorts, target_jpos_urdf_raw):
    """
    Sends joint commands to the serial port by decoding the joint positions and sending them in the correct format.

    Args:
    - goodPorts: Dictionary of connected serial ports.
    - target_jpos_urdf_raw: A NumPy array representing the target joint positions in URDF format (in radians).
    """
    try:
        # Decode the joint positions into the required format for sending
        
        task_inst = decode_jpos_i_command(target_jpos_urdf_raw)
        # Create the task with the decoded joint positions
        task1 = ['I', task_inst, 0.0]  # Sending with delay 0.00 (adjust if necessary)
        send(goodPorts, task1)
        # print(f"Decoded Joint Command: {task_inst}")

        # time.sleep(0.02)


        # Send the task through the serial port
        # return_i = send(goodPorts, task1)
        # print(f"Response from Serial: {return_i}")

    except Exception as e:
        print(f"Error in sending joint command: {e}")


def require_gyro_and_dof_data(goodPorts):
    """
    This function sends serial commands to request gyro, linear acceleration, and DOF positions data.
    It parses and decodes the results for further use.

    Args:
    - goodPorts: Dictionary of connected serial ports.

    Returns:
    - lin_acc: Linear acceleration from the decoded data.
    - ang_vel: Angular velocity from the decoded data.
    - dof_positions: DOF positions from the decoded data.
    """
    try:
        # Request gyro and linear acceleration data
        return_v = send(goodPorts, ['v', 0.00])
        lin_acc, ang_vel = decode_gyro_result(return_v)
        print(f"Linear Acceleration: {lin_acc}, Angular Velocity: {ang_vel}")
    except Exception as e:
        print(f"Error reading gyro and acceleration data: {e}")
        lin_acc, ang_vel = None, None

    try:
        # Request joint DOF position data
        return_j = send(goodPorts, ['j', 0.00])  # Send command to get joint positions
        dof_positions = decode_jpos_result(return_j)
        print(f"DOF Positions: {dof_positions}")
    except Exception as e:
        print(f"Error reading joint DOF data: {e}")
        dof_positions = None

    return lin_acc, ang_vel, dof_positions
