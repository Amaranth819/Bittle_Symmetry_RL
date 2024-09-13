import numpy as np
import numpy.linalg as la
import bittle_transformations as tr
import math

class ImuParameters:
    def __init__(self):
        self.frequency = 200
        self.sigma_a_n = 0.0     # acc noise.   m/(s*sqrt(s)), continuous noise sigma
        self.sigma_w_n = 0.0     # gyro noise.  rad/sqrt(s), continuous noise sigma
        self.sigma_a_b = 0.0     # acc bias     m/sqrt(s^5), continuous bias sigma
        self.sigma_w_b = 0.0     # gyro bias    rad/sqrt(s^3), continuous bias sigma

class ESEKF(object):
    def __init__(self, init_nominal_state: np.array, imu_parameters: ImuParameters):
        """
        :param init_nominal_state: [ p, q, v, a_b, w_b, g ], a 19x1 or 1x19 vector
        :param imu_parameters: imu parameters
        """
        self.nominal_state = init_nominal_state
        if self.nominal_state[3] < 0:
            self.nominal_state[3:7] *= 1    # force the quaternion has a positive real part.
        self.imu_parameters = imu_parameters

        # initialize noise covariance matrix
        noise_covar = np.zeros((12, 12))
        # assume the noises (especially sigma_a_n) are isotropic so that we can precompute self.noise_covar and save it.
        noise_covar[0:3, 0:3] = (imu_parameters.sigma_a_n**2) * np.eye(3)
        noise_covar[3:6, 3:6] = (imu_parameters.sigma_w_n**2) * np.eye(3)
        noise_covar[6:9, 6:9] = (imu_parameters.sigma_a_b**2) * np.eye(3)
        noise_covar[9:12, 9:12] = (imu_parameters.sigma_w_b**2) * np.eye(3)
        
        G = np.zeros((18, 12))
        G[3:6, 3:6] = -np.eye(3)
        G[6:9, 0:3] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        self.noise_covar = G @ noise_covar @ G.T

        # initialize error covariance matrix
        self.error_covar = 0.01 * self.noise_covar

        self.last_predict_time = 0.0

    def predict(self, imu_measurement: np.array):
        """
        :param imu_measurement: [t, w_m, a_m]
        :return: 
        """
        if self.last_predict_time == imu_measurement[0]:
            return
        # we predict error_covar first, because __predict_nominal_state will change the nominal state.
        self.__predict_error_covar(imu_measurement)
        self.__predict_nominal_state(imu_measurement)
        self.last_predict_time = imu_measurement[0]  # update timestamp

    def update(self, gt_measurement: np.array, measurement_covar: np.array):
        """
        :param gt_measurement: [p, q], a 7x1 or 1x7 vector
        :param measurement_covar: a 6x6 symmetrical matrix = diag{sigma_p^2, sigma_theta^2}
        :return: 
        """
        H = np.zeros((6, 18))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        PHt = self.error_covar @ H.T  # 18x6
        K = PHt @ la.inv(H @ PHt + measurement_covar)  # 18x6

        self.error_covar = (np.eye(18) - K @ H) @ self.error_covar
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

        if gt_measurement[3] < 0:
            gt_measurement[3:7] *= -1
        gt_p = gt_measurement[0:3]
        gt_q = gt_measurement[3:7]
        q = self.nominal_state[3:7]

        delta = np.zeros((6, 1))
        delta[0:3, 0] = gt_p - self.nominal_state[0:3]
        delta_q = tr.quaternion_multiply(tr.quaternion_conjugate(q), gt_q)
        if delta_q[0] < 0:
            delta_q *= -1
        angle = math.asin(la.norm(delta_q[1:4]))
        if math.isclose(angle, 0):
            axis = np.zeros(3,)
        else:
            axis = delta_q[1:4] / la.norm(delta_q[1:4])
        delta[3:6, 0] = angle * axis

        errors = K @ delta

        self.nominal_state[0:3] += errors[0:3, 0]  # update position
        dq = tr.quaternion_about_axis(la.norm(errors[3:6, 0]), errors[3:6, 0])
        self.nominal_state[3:7] = tr.quaternion_multiply(q, dq)  # update rotation
        self.nominal_state[3:7] /= la.norm(self.nominal_state[3:7])
        if self.nominal_state[3] < 0:
            self.nominal_state[3:7] *= 1
        self.nominal_state[7:] += errors[6:, 0]  # update the rest.

        G = np.eye(18)
        G[3:6, 3:6] = np.eye(3) - tr.skew_matrix(0.5 * errors[3:6, 0])
        self.error_covar = G @ self.error_covar @ G.T


    def __predict_nominal_state(self, imu_measurement: np.array):
        p = self.nominal_state[:3].reshape(-1, 1)
        q = self.nominal_state[3:7]
        v = self.nominal_state[7:10].reshape(-1, 1)
        a_b = self.nominal_state[10:13].reshape(-1, 1)
        w_b = self.nominal_state[13:16]
        g = self.nominal_state[16:19].reshape(-1, 1)

        w_m = imu_measurement[1:4].copy()
        a_m = imu_measurement[4:7].reshape(-1, 1).copy()
        dt = imu_measurement[0] - self.last_predict_time

        w_m -= w_b
        a_m -= a_b

        angle = la.norm(w_m)
        axis = w_m / angle
        R_w = tr.rotation_matrix(0.5 * dt * angle, axis)
        q_w = tr.quaternion_from_matrix(R_w, True)
        q_half_next = tr.quaternion_multiply(q, q_w)

        R_w = tr.rotation_matrix(dt * angle, axis)

    # Function to extract state information
    def extract_state_information(self, predicted_state, imu_angular_velocity):
        quaternion = predicted_state[3:7]  # Quaternion: [q_w, q_x, q_y, q_z]
        r = tr.quaternion_matrix(quaternion)[:3, :3]  # Extract rotation matrix from quaternion

        global_gravity = np.array([0, 0, 9.81])  # Gravity in world frame
        projected_gravity = r @ global_gravity  # Rotate gravity into body frame

        linear_velocity = predicted_state[7:10]  # Linear velocity
        gyroscope_bias = predicted_state[13:16]  # Gyroscope bias

        angular_velocity = imu_angular_velocity - gyroscope_bias  # Adjust angular velocity by removing bias
        return projected_gravity, linear_velocity, angular_velocity
