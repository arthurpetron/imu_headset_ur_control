import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simple_head_simulator import SimpleHeadSimulator

# Generate test data
simulator = SimpleHeadSimulator(I=None, k=None, gamma=None)
duration = 5.0
dt = 0.01
timestamps = np.arange(0, duration, dt)
orientations = []
residuals = []
gyros = []
accels = []

for t in timestamps:
    gyro = np.array([0.1 * np.sin(0.5 * t), 0.05 * np.cos(0.3 * t), 0.02])  # fake gyro
    accel = simulator.estimate_accel() + np.random.normal(0, 0.01, 3)       # fake accel with noise

    simulator.apply_imu(gyro, accel, simulator.last_update + dt)
    
    orientations.append(simulator.get_orientation_euler())
    residuals.append(accel - simulator.estimate_accel())
    gyros.append(gyro)
    accels.append(accel)

# Convert to arrays
orientations = np.array(orientations)
residuals = np.array(residuals)
gyros = np.array(gyros)
accels = np.array(accels)

# Plotting
sns.set_theme(style="whitegrid")
time = timestamps

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Head orientation
axes[0].plot(time, orientations[:, 0], label='Pitch (deg)')
axes[0].plot(time, orientations[:, 1], label='Yaw (deg)')
axes[0].plot(time, orientations[:, 2], label='Roll (deg)')
axes[0].set_ylabel("Orientation (°)")
axes[0].legend()
axes[0].set_title("Simulated Head Orientation Over Time")

# IMU Gyroscope
axes[1].plot(time, gyros[:, 0], label='Gyro X (rad/s)')
axes[1].plot(time, gyros[:, 1], label='Gyro Y (rad/s)')
axes[1].plot(time, gyros[:, 2], label='Gyro Z (rad/s)')
axes[1].set_ylabel("Gyro (rad/s)")
axes[1].legend()
axes[1].set_title("IMU Gyroscope Input")

# IMU Accelerometer
axes[2].plot(time, accels[:, 0], label='Accel X (m/s²)')
axes[2].plot(time, accels[:, 1], label='Accel Y (m/s²)')
axes[2].plot(time, accels[:, 2], label='Accel Z (m/s²)')
axes[2].set_ylabel("Accel (m/s²)")
axes[2].legend()
axes[2].set_title("IMU Accelerometer Input")

# Residuals
axes[3].plot(time, residuals[:, 0], label='Residual X')
axes[3].plot(time, residuals[:, 1], label='Residual Y')
axes[3].plot(time, residuals[:, 2], label='Residual Z')
axes[3].set_ylabel("Accel Residual")
axes[3].set_xlabel("Time (s)")
axes[3].legend()
axes[3].set_title("Accelerometer Residuals (Measured - Modeled)")

plt.tight_layout()
plt.show()