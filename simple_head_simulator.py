import threading
import numpy as np
import time

class SimpleHeadSimulator:
    def __init__(self, I: np.ndarray, k: np.ndarray, gamma: np.ndarray, dt=0.01):
        """
        I, k, gamma are 3D vectors (x, y, z) for inertia, stiffness, and damping
        """
        self.lock = threading.Lock()

        if (I == None or k is None or gamma is None):
            I, k, gamma = self.estimate_I_k_gamma(self)

        self.I = I
        self.k = k
        self.gamma = gamma
        self.dt = dt

        self.x = np.zeros(6)  # state vector: [theta_x, y, z, omega_x, y, z]
        self.P = np.eye(6) * 0.01  # covariance matrix
        self.Q = np.eye(6) * 1e-4  # process noise
        self.R = np.eye(3) * 1e-2  # measurement noise

        self.r_imu = np.array([0.0, 0.0, 0.1])  # IMU position relative to neck pivot in meters
        self.torque_input = np.zeros(3)
        self.last_update = time.time()

    # Estimate head inertia based on head circumference and length
    # From from Richard N. Hinrichs. Regression equations to predict segmental moments of
    # inertia from anthropometric measurements: An extension of the data of Chandler
    # et al (1975). J. Biomechanics, 18(8):621-624, 1985.
    @staticmethod
    def estimate_head_inertia(headc_mm=570, headl_mm=190):
        return 25.102 * headc_mm - 6.4805 * headl_mm - 1122.6  # in kg·mm²
    
    @staticmethod
    def estimate_I_k_gamma(self):
        I_scalar = self.estimate_head_inertia()
        I = np.array([I_scalar, I_scalar, I_scalar]) / 1e6  # convert to kg·m²
        k = np.array([1.5, 2.0, 1.8])   # stiffness per axis (N·m/rad)
        gamma = np.array([0.2, 0.3, 0.25])  # damping (N·m·s/rad)

        return I, k, gamma
    
    def apply_imu(self, gyro: np.ndarray, accel: np.ndarray, timestamp: float):
        with self.lock:
            dt = timestamp - self.last_update
            if dt <= 0: return
            self.torque_input = self.estimate_torque(gyro)
            self.integrate(dt)
            self.last_update = timestamp

            self.kalman_update(accel)

    def estimate_torque(self, gyro: np.ndarray) -> np.ndarray:
        return self.I * (gyro - self.x[3:6]) / self.dt
    
    def estimate_accel(self, x: np.ndarray = None) -> np.ndarray:
        if x is None:
            x = self.x
        theta = x[0:3]
        omega = x[3:6]
        r = self.r_imu
        alpha = (self.torque_input - self.gamma * omega - self.k * theta) / self.I
        term1 = np.cross(alpha, r)
        term2 = np.cross(omega, np.cross(omega, r))
        return term1 + term2

    def kalman_update(self, accel_meas):
        x = self.x

        # Predict modeled accel from x
        accel_pred = self.estimate_accel(self.x)

        # Measurement residual
        y = accel_meas - accel_pred  # Innovation

        # Compute Jacobian H of accel w.r.t. state x
        H = self.compute_accel_jacobian(x)

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        # Update internal state
        self.theta = self.x[0:3]
        self.omega = self.x[3:6]

    def compute_accel_jacobian(self, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Computes the Jacobian of the estimated accelerometer reading w.r.t. state vector x.
        x: current state [theta_x, theta_y, theta_z, omega_x, omega_y, omega_z]
        Returns: H, shape (3, 6), where H[i,j] = ∂a_i / ∂x_j
        """
        H = np.zeros((3, 6))

        for i in range(6):
            x_perturbed = x.copy()
            x_perturbed[i] += epsilon

            # Compute perturbed state
            a_perturbed = self.estimate_accel(x_perturbed)

            # Compute derivative
            a_nominal = self.estimate_accel(x)
            H[:, i] = (a_perturbed - a_nominal) / epsilon

        return H
    
    def integrate(self, dt: float):
        theta = self.x[0:3]
        omega = self.x[3:6]
        alpha = (self.torque_input - self.gamma * omega - self.k * theta) / self.I
        omega += alpha * dt
        theta += omega * dt
        theta = np.clip(theta, np.radians([-40, -55, -70]), np.radians([40, 45, 70]))
        self.x[0:3] = theta
        self.x[3:6] = omega

    def get_orientation_euler(self) -> np.ndarray:
        with self.lock:
            return np.degrees(self.x[0:3].copy())

    def simulate_forward(self):
        with self.lock:
            now = time.time()
            dt = now - self.last_update
            if dt > 0:
                self.integrate(dt)
                self.last_update = now