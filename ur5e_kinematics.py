import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# UR5e Denavit–Hartenberg parameters: [theta offset, d, a, alpha]
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

def ik_objective(joint_angles, desired_tcp):
    T = forward_kinematics(joint_angles)
    return np.linalg.norm(T[:3, 3] - desired_tcp)

def inverse_kinematics(desired_tcp, initial_guess=None):
    if initial_guess is None:
        initial_guess = np.zeros(6)
    result = minimize(ik_objective, initial_guess, args=(desired_tcp,), bounds=[(-np.pi, np.pi)]*6)
    if result.success:
        return result.x
    else:
        raise RuntimeError("Inverse kinematics failed to converge.")