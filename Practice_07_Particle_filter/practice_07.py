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

# ============================================================================
# Practice 2: Particle Filter with Dead Reckoning Model
# ============================================================================
# Task: Implement particle filter for 2D vehicle state estimation
# Process model: Dead reckoning (same as Practice 1)
# Measurement model: Bearing sensor (range and angle)
# Goal: Estimate states (x, y, psi) and compare with true values
# ============================================================================

n_particles = 500  # Number of particles

# Process noise covariance Q (tuning parameter)
# Q represents uncertainty in the motion model
Q = np.diag([0.5**2, 0.5**2, np.radians(1.0)**2])  # [m^2, m^2, rad^2]

# Measurement noise coviariance R
# R should match the sensor noise from Practice 1-2
R = np.diag([std_r**2, std_theta**2])  # [m^2, rad^2]

# Initialize particles around initial state with some spread
particles = np.zeros((n_particles, 3))  # Each row: [x, y, psi]
particles[:, 0] = x_true[0] + np.random.normal(0, 0.5, n_particles)  # x
particles[:, 1] = y_true[0] + np.random.normal(0, 0.5, n_particles)  # y
particles[:, 2] = psi_true[0] + np.random.normal(0, np.radians(5.0), n_particles)  # psi

# Initialize weights uniformly
weights = np.ones(n_particles) / n_particles

# Storage for estimated states
x_est = np.zeros(N)
y_est = np.zeros(N)
psi_est = np.zeros(N)

# Initial estimate (weighted mean of particles)
x_est[0] = np.sum(weights * particles[:, 0])
y_est[0] = np.sum(weights * particles[:, 1])
psi_est[0] = np.sum(weights * particles[:, 2])

# Particle filter helper functions

def predict_particles(particles, V, psi_dot, dt, Q):
    n_particles = particles.shape[0]

    # Gnerate process noise for each particle
    process_noise = np.random.multivariate_normal(np.zeros(3), Q, n_particles)

    # Apply motion model to each particle
    particles_new = np.copy(particles)
    particles_new[:, 0] += V * dt * np.cos(particles[:, 2]) + process_noise[:, 0]  # x
    particles_new[:, 1] += V * dt * np.sin(particles[:, 2]) + process_noise[:, 1]  # y
    particles_new[:, 2] += psi_dot * dt + process_noise[:, 2]  # psi

    return particles_new

def measurement_model(particles):
    x = particles[:, 0]
    y = particles[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return np.column_stack((r, theta))

def update_weights(particles, weights, z_meas, R):
    # Compute expected measurements for all particles
    z_pred = measurement_model(particles)

    # compute innovation (measurement residual)
    innovation = z_meas - z_pred

    # Handle angle wrapping for theta
    innovation[:, 1] = np.arctan2(np.sin(innovation[:, 1]), np.cos(innovation[:, 1]))

    # Compute likelihood for each particle using multivariate normal distribution
    R_inv = np.linalg.inv(R)

    weights_new = np.zeros(len(weights))
    for i in range(len(particles)):
        innov = innovation[i, :]
        exponent = -0.5 * innov.T @ R_inv @ innov
        weights_new[i] = weights[i] * np.exp(exponent) 

    # Normalize weights
    weights_sum = np.sum(weights_new)
    if weights_sum > 0: 
        weights_new /= weights_sum
    else:
        weights_new = np.ones(len(weights)) / len(weights)  # Avoid division by zero

    return weights_new

def resample_particles(particles, weights):
    n_particles = len(weights)

    # Compute cumulative sum of weights
    cumsum = np.cumsum(weights)

    # Generate systematic samples
    u = (np.arange(n_particles) + np.random.uniform()) / n_particles

    # Resample
    indices = np.searchsorted(cumsum, u)
    particles_new = particles[indices]

    # Reset weights to uniform
    weights_new = np.ones(n_particles) / n_particles
    
    return particles_new, weights_new

def effective_sample_size(weights):
    # Compute effective sample size
    # ESS = 1 / sum(w_i^2)
    return 1.0 / np.sum(weights**2)


# Particle filter main loop

resample_threshold = n_particles / 2.0  # Resample when ESS < threshold

for k in range(1, N):
    # Prediction step
    particles = predict_particles(particles, velocity[k-1], yaw_rate[k-1], dt, Q)

    # Measurement update step
    z_meas = np.array([r_meas[k], theta_meas[k]])
    weights = update_weights(particles, weights, z_meas, R)

    # Resampling step
    ESS = effective_sample_size(weights)
    if ESS < resample_threshold:
        particles, weights = resample_particles(particles, weights)

    # State estimation (weighted mean)
    x_est[k] = np.sum(weights * particles[:, 0])
    y_est[k] = np.sum(weights * particles[:, 1])
    psi_est[k] = np.sum(weights * particles[:, 2])


# ============================================================================
# Practice 2-1: Plot Estimated States and Errors
# ============================================================================

# Compute estimation errors
error_x = x_est - x_true
error_y = y_est - y_true
error_psi = psi_est - psi_true
# Handle angle wrapping for heading error
error_psi = np.arctan2(np.sin(error_psi), np.cos(error_psi))

# Plot estimated states vs true states
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# X position
axes[0, 0].plot(time, x_true, 'b-', linewidth=2, label='True')
axes[0, 0].plot(time, x_est, 'r--', linewidth=2, label='Estimated')
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('X Position [m]')
axes[0, 0].set_title('X Position Estimation')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(time, error_x, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Time [s]')
axes[0, 1].set_ylabel('Error [m]')
axes[0, 1].set_title('X Position Error')
axes[0, 1].grid(True)

# Y position
axes[1, 0].plot(time, y_true, 'b-', linewidth=2, label='True')
axes[1, 0].plot(time, y_est, 'r--', linewidth=2, label='Estimated')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Y Position [m]')
axes[1, 0].set_title('Y Position Estimation')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(time, error_y, 'g-', linewidth=2)
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_ylabel('Error [m]')
axes[1, 1].set_title('Y Position Error')
axes[1, 1].grid(True)

# Heading
axes[2, 0].plot(time, np.degrees(psi_true), 'b-', linewidth=2, label='True')
axes[2, 0].plot(time, np.degrees(psi_est), 'r--', linewidth=2, label='Estimated')
axes[2, 0].set_xlabel('Time [s]')
axes[2, 0].set_ylabel('Heading [deg]')
axes[2, 0].set_title('Heading Estimation')
axes[2, 0].legend()
axes[2, 0].grid(True)

axes[2, 1].plot(time, np.degrees(error_psi), 'g-', linewidth=2)
axes[2, 1].set_xlabel('Time [s]')
axes[2, 1].set_ylabel('Error [deg]')
axes[2, 1].set_title('Heading Error')
axes[2, 1].grid(True)

plt.tight_layout()
plt.savefig('practice_2_1_estimation.png', dpi=300)
plt.show()

# Plot 2D trajectory comparison
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x_true, y_true, 'b-', linewidth=2, label='True Trajectory')
ax.plot(x_est, y_est, 'r--', linewidth=2, label='Estimated Trajectory')
ax.plot(x_true[0], y_true[0], 'go', markersize=10, label='Start')
ax.plot(0, 0, 'ks', markersize=12, label='Sensor')
ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.set_title('2D Trajectory Comparison')
ax.legend()
ax.grid(True)
ax.axis('equal')
plt.tight_layout()
plt.savefig('practice_2_1_trajectory.png', dpi=300)
plt.show()

# ============================================================================
# Practice 2-2: Compute RMSE
# ============================================================================

# Calculate Root Mean Square Error for each state
rmse_x = np.sqrt(np.mean(error_x**2))
rmse_y = np.sqrt(np.mean(error_y**2))
rmse_psi = np.sqrt(np.mean(error_psi**2))

print("\n" + "="*60)
print("Practice 2-2: RMSE Results")
print("="*60)
print(f"RMSE X:       {rmse_x:.4f} m")
print(f"RMSE Y:       {rmse_y:.4f} m")
print(f"RMSE Heading: {np.degrees(rmse_psi):.4f} deg ({rmse_psi:.4f} rad)")
print(f"\nProcess Noise Q diagonal: {np.sqrt(np.diag(Q))}")
print(f"Measurement Noise R diagonal: {np.sqrt(np.diag(R))}")
print("="*60)