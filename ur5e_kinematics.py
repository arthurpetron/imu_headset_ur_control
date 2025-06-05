# ur5e_kinematics.py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# UR5e Denavit–Hartenberg parameters
DH_PARAMS = [
    [0,     0.1625,   0.0,      np.pi/2],
    [0,     0.0,      -0.425,   0.0],
    [0,     0.0,      -0.3922,  0.0],
    [0,     0.1333,   0.0,      np.pi/2],
    [0,     0.0997,   0.0,     -np.pi/2],
    [0,     0.0996,   0.0,      0.0]
]

def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1]
    ])

def forward_kinematics(joint_angles):
    T = np.eye(4)
    for i in range(6):
        theta = joint_angles[i] + DH_PARAMS[i][0]
        d, a, alpha = DH_PARAMS[i][1:]
        T = T @ dh_transform(theta, d, a, alpha)
    return T

def ik_objective(joint_angles, desired_pos, desired_rotmat, pos_weight=1.0, rot_weight=1.0):
    T = forward_kinematics(joint_angles)
    pos_err = np.linalg.norm(T[:3, 3] - desired_pos)
    rot_err = np.linalg.norm(T[:3, :3] - desired_rotmat)
    return pos_weight * pos_err**2 + rot_weight * rot_err**2

def inverse_kinematics(desired_pos, desired_quat, initial_guess=None):
    if initial_guess is None:
        initial_guess = np.zeros(6)
    desired_rotmat = R.from_quat(desired_quat).as_matrix()
    result = minimize(
        ik_objective,
        initial_guess,
        args=(desired_pos, desired_rotmat),
        bounds=[(-np.pi, np.pi)] * 6,
        options={'maxiter': 200}
    )
    if result.success:
        return result.x
    else:
        raise RuntimeError("Inverse kinematics failed to converge.")
