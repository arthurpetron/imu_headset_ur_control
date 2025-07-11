# ur54e_control_interface.py
import numpy as np
from scipy.spatial.transform import Rotation as R
from ur5e_kinematics import inverse_kinematics

# UR5e Head Control Interface for ezmsg - https://github.com/ezmsg-org/ezmsg
# This interface computes the TCP position and orientation based on head orientation
# from a simulated or real IMU. It uses a spherical coordinate system to
# determine the position of the TCP (Tool Center Point) based on pitch, yaw, and roll angles
# and the joint 6 angle (yaw) for the UR5e robot arm.

class UR5eControlInterface:
    def __init__(self,
                 r_min=0.85 * 0.5,
                 r_max=0.85 * 0.9,
                 r_default=0.85 * 0.75,
                 max_angle=np.radians(90),
                 yaw_gain=1.0):
        self.r_min = r_min
        self.r_max = r_max
        self.r_default = r_default
        self.max_angle = max_angle
        self.yaw_gain = yaw_gain
        self.radius = r_default
        self.tcp_position = np.zeros(3)
        self.j6_angle = 0.0

    def update_from_head_orientation(self, theta: np.ndarray):
        pitch, yaw, roll = np.clip(theta, -self.max_angle, self.max_angle)
        self.radius = np.clip(self.radius, self.r_min, self.r_max)
        x = self.radius * np.sin(pitch)
        y = self.radius * np.sin(roll)
        z = self.radius * np.sqrt(max(0.0, 1 - np.sin(pitch)**2 - np.sin(roll)**2))
        self.tcp_position = np.array([x, y, z])
        self.j6_angle = self.yaw_gain * yaw
        return self.tcp_position, self.j6_angle

    def compute_tcp_orientation(self):
        z_axis = self.tcp_position / np.linalg.norm(self.tcp_position)
        up_hint = np.array([0.0, 0.0, 1.0])
        if np.allclose(z_axis, up_hint):
            up_hint = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up_hint, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        return R.from_matrix(rot_matrix).as_quat()
    
    def compute_joint_angles(self):
        tcp_orientation = self.compute_tcp_orientation()
        return inverse_kinematics(self.tcp_position, tcp_orientation)
        
        