import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Practice 1: True State and Measurement Simulation for 2D Vehicle
# ============================================================================
# Problem Description:
# - Simulate 2D vehicle motion (x, y, heading) using dead reckoning model
# - Simulate bearing sensor measurements (range r, angle theta) from origin
# - Motion model: x[k+1] = x[k] + V*dt*cos(psi[k])
#                 y[k+1] = y[k] + V*dt*sin(psi[k])
#                 psi[k+1] = psi[k] + dt*psi_dot
# - Measurement model: r = sqrt(x^2 + y^2)
#                      theta = atan2(y, x)
# ============================================================================

# ============================================================================
# Practice 1-1: True State Simulation
# ============================================================================
# Task: Simulate vehicle states (x, y, psi) from t=0 to t=50
# Given: Velocity and yaw rate as sinusoidal inputs
# Output: Plot each state vs time and 2D trajectory
# ============================================================================

# Simulation parameters
dt = 0.1  # Sampling time [s]
t_end = 50  # Simulation end time [s]
time = np.arange(0, t_end + dt, dt)
N = len(time)

# Initial conditions
x_init = 1.0  # Initial x position [m]
y_init = 1.0  # Initial y position [m]
psi_init = np.radians(45)  # Initial heading [rad]

# Velocity input: V(t) = 3*sin(2*pi*t/(2*pi)) + 6
velocity_period = 2 * np.pi
velocity_amplitude = 3.0
velocity_shift = 6.0

# Yaw rate input: psi_dot(t) = (6*pi/180)*sin(2*pi*t/(6*pi)) + (10*pi/180)
yaw_rate_period = 6 * np.pi
yaw_rate_amplitude = 6 * np.pi / 180
yaw_rate_shift = 10 * np.pi / 180

# Generate input signals
velocity = velocity_amplitude * np.sin(2 * np.pi * time / velocity_period) + velocity_shift # [m/s]
yaw_rate = yaw_rate_shift + yaw_rate_amplitude * np.sin(2 * np.pi * time / yaw_rate_period)  # [rad/s]

# Initialize state arrays
x_true = np.zeros(N)
y_true = np.zeros(N)
psi_true = np.zeros(N)

x_true[0] = x_init
y_true[0] = y_init
psi_true[0] = psi_init

# Dead reckoning model simulation
for k in range(N - 1):
    x_true[k + 1] = x_true[k] + velocity[k] * dt * np.cos(psi_true[k])
    y_true[k + 1] = y_true[k] + velocity[k] * dt * np.sin(psi_true[k])
    psi_true[k + 1] = psi_true[k] + dt * yaw_rate[k]

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(time, x_true, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('X Position [m]')
axes[0, 0].set_title('X Position over Time')
axes[0, 0].grid(True)

axes[0, 1].plot(time, y_true, 'r-', linewidth=2)
axes[0, 1].set_xlabel('Time [s]')
axes[0, 1].set_ylabel('Y Position [m]')
axes[0, 1].set_title('Y Position over Time')
axes[0, 1].grid(True)

axes[1, 0].plot(time, np.degrees(psi_true), 'g-', linewidth=2)
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Heading [deg]')
axes[1, 0].set_title('Heading Angle over Time')
axes[1, 0].grid(True)

axes[1, 1].plot(x_true, y_true, 'b-', linewidth=2, label='Trajectory')
axes[1, 1].plot(x_true[0], y_true[0], 'go', markersize=10, label='Start')
axes[1, 1].plot(x_true[-1], y_true[-1], 'ro', markersize=10, label='End')
axes[1, 1].plot(0, 0, 'ks', markersize=12, label='Sensor')
axes[1, 1].set_xlabel('X Position [m]')
axes[1, 1].set_ylabel('Y Position [m]')
axes[1, 1].set_title('2D Vehicle Trajectory')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].axis('equal')

plt.tight_layout()
plt.savefig('practice_1_1_states.png', dpi=300)
plt.show()

# ============================================================================
# Practice 1-2: Measurement Simulation with Sensor Noise
# ============================================================================
# Task: Simulate bearing sensor measurements with Gaussian noise
# Sensor noise: std_r = 1.0 m, std_theta = 3.0 deg
# Measurement model: r = sqrt(x^2 + y^2) + noise_r
#                    theta = atan2(y, x) + noise_theta
# Output: Plot true vs noisy measurements, save to file
# ============================================================================

# Sensor noise standard deviations
std_r = 1.0 # Range noise std [m]
std_theta = np.radians(3.0) # Angle noise std [rad]

# Compute true measurements (noiseless)
r_true = np.sqrt(x_true**2 + y_true**2)
theta_true = np.arctan2(y_true, x_true)

# Generate noisy measurements
np.random.seed(42)  # For reproducibility
noise_r = np.random.normal(0, std_r, N)
noise_theta = np.random.normal(0, std_theta, N)

r_meas = r_true + noise_r
theta_meas = theta_true + noise_theta

# Plot measurements
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(time, r_true, 'b-', linewidth=2, label='True Range')
axes[0].plot(time, r_meas, 'r.', markersize=4, alpha=0.5, label='Measured Range')
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Range [m]')
axes[0].set_title('Range Measurement (std = 1.0 m)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time, np.degrees(theta_true), 'b-', linewidth=2, label='True Angle')
axes[1].plot(time, np.degrees(theta_meas), 'r.', markersize=4, alpha=0.5, label='Measured Angle')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Angle [deg]')
axes[1].set_title('Angle Measurement (std = 3.0 deg)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('practice_1_2_measurements.png', dpi=300)
plt.show()
