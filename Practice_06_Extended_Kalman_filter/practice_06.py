"""
-------------------------------------------------------------------------------
Problem: Practice 1-1: True state simulation for 2D vehicle
Based on: "Optimal State Estimation" by Dan Simon & Practice_06 PDF
-------------------------------------------------------------------------------

1. 목표 (Objective):
   - 주어진 속도(Velocity)와 각속도(Yaw rate) 입력 데이터를 사용하여
     0초부터 30초까지 차량의 2D 궤적(x, y) 및 헤딩(psi)을 시뮬레이션함.

2. 시스템 모델 (System Model - Dead Reckoning):
   상태 변수 vector x = [x_pos, y_pos, psi]^T 에 대하여:
   x_{t+1}   = x_t + V * dT * cos(psi_t)
   y_{t+1}   = y_t + V * dT * sin(psi_t)
   psi_{t+1} = psi_t + dT * yaw_rate_t

3. 초기 조건 (Initial Conditions):
   - Sampling time (dT) = 0.1 sec
   - Total time = 30 sec
   - Initial Position: x = 1m, y = 1m
   - Initial Heading: psi = 45 degrees

4. 입력 데이터 (Input Data Profiles) (Based on User Query):
   (1) Velocity (V):
       - Period: 2π, Amplitude: 3, Vertical Shift: 6
       - Formula: V(t) = 3 * sin(t) + 6
   (2) Yaw Rate (psi_dot):
       - Period: 2π, Amplitude: π/18, Vertical Shift: 0
       - Formula: psi_dot(t) = (π/18) * cos(t)

-------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 시뮬레이션 설정
dT = 0.1  # 샘플링 시간 (초)
total_time = 30  # 총 시뮬레이션 시간 (초)
time = np.arange(0, total_time, dT)  # 시간 벡터
n_steps = len(time)  # 시뮬레이션 스텝 수

# 2. 입력 데이터 생성 
# Velocity: V = 3*sin(t) + 6
input_v = 3 * np.sin(time) + 6

# Yaw Rate: psi_dot = (π/18)*cos(t)
# 주파수(omega) = 2pi / 6pi = 1/3
input_yawrate = (np.pi / 18) * np.cos(time)

# 3. 상태 변수 초기화
# 상태 벡터 x = [x_pos, y_pos, heading_angle]
x = np.zeros(n_steps)
y = np.zeros(n_steps)
psi = np.zeros(n_steps)

# 초기값 설정
x[0] = 1.0     # initial x = 1m
y[0] = 1.0     # initial y = 1m
psi[0] = np.deg2rad(45)  # initial heading = 45 degrees in radians

# 4. Dead Reackoning 시뮬레이션
# f(x, u, w, t)
for t in range(n_steps - 1):
    # 현재 상태 및 입력
    current_psi = psi[t]
    v_t = input_v[t]
    yawrate_t = input_yawrate[t]

    # 다음 상태 계선 (비선형 모델)
    x[t + 1] = x[t] + v_t * dT * np.cos(current_psi)
    y[t + 1] = y[t] + v_t * dT * np.sin(current_psi)
    psi[t + 1] = psi[t] + dT * yawrate_t

# 5. 결과 시각화
plt.figure(figsize=(12, 10))

# 입력 데이터 확인 - Velocity
plt.subplot(3, 2, 1)
plt.plot(time, input_v, 'b-', linewidth=1.5)
plt.title('Input Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)

# 입력 데이터 확인 - Yaw Rate
plt.subplot(3, 2, 2)
plt.plot(time, input_yawrate, 'r-', linewidth=1.5)
plt.title('Input Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('Yaw Rate (rad/s)')
plt.grid(True)

# 상태 변수 - 위치 (x, y)
plt.subplot(3, 2, 3)
plt.plot(time, x, 'b-', linewidth=1.5, label='X Position')
plt.title('Position (X)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(time, y, 'r-', linewidth=1.5, label='Y Position')
plt.title('Position (Y)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

# 헤딩 각도 (Psi)
plt.subplot(3, 2, 5)
plt.plot(time, np.rad2deg(psi))
plt.title('Heading Angle (deg)')
plt.xlabel('Time (s)')
plt.ylabel('Degree')
plt.grid(True)

# 2D 궤적 (Trajectory)
plt.subplot(3, 2, 6)
plt.plot(x, y, 'b-', linewidth=2)
plt.plot(x, y, 'ro', label='Start') # 시작점 표시
plt.plot(x[-1], y[-1], 'go', label='End') # 끝점 표시
plt.title('2D Vehicle Trajectory (Dead Reckoning)')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.axis('equal') # X, Y 축 비율 동일하게
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


"""
-------------------------------------------------------------------------------
Problem: Practice 1-2: Measurement Simulation
Based on: "Optimal State Estimation" by Dan Simon & Practice_06 PDF
-------------------------------------------------------------------------------

1. 목표 (Objective):
   - Practice 1-1에서 생성한 참값(True State) 위치 데이터(x, y)를 사용하여
     센서 측정값(Measurement)을 시뮬레이션함.
   - 측정값에 가우시안 노이즈(Gaussian Noise)를 추가하여 현실적인 센서 데이터를 생성.

2. 측정 모델 (Measurement Model):
   원점(0, 0)에 위치한 베어링 센서(Bearing Sensor)가 차량의 거리와 각도를 측정함.
   측정 벡터 y_t = [r, theta]^T
   - 거리 (Range, r): r = sqrt(x^2 + y^2)
   - 방위각 (Bearing, theta): theta = atan2(y, x)

3. 센서 노이즈 조건 (Sensor Noise Parameters):
   측정값에는 평균이 0인 가우시안 노이즈(v)가 추가됨. y_t = h(x_t) + v_t
   (1) 거리 노이즈 표준편차 (Sigma_r): 1.0 m
   (2) 방위각 노이즈 표준편차 (Sigma_theta): 3 degrees
       (* 주의: 계산 시 라디안 단위로 변환 필요: 3 * pi / 180)

4. 출력 데이터 (Output Data):
   - Noisy Range (r_measure)
   - Noisy Bearing (theta_measure)
-------------------------------------------------------------------------------
"""

# 1. 센서 노이즈 설정 (Sensor Noise Parameters)
sigma_r = 1.0  # Range standard deviation = 1.0 m
sigma_theta_deg = 3.0  # Bearing standard deviation = 3 degrees
sigma_theta_rad = np.deg2rad(sigma_theta_deg)  # Convert to radians

# 2. 참값 측정치 계선 (True Measurement Calculation)
# 거리 r = sqrt(x^2 + y^2)
true_r = np.sqrt(x**2 + y**2)

# 방위각 theta = atan2(y, x)
true_theta = np.arctan2(y, x)

# 3. 측정 노이즈 생성 (Generate Gaussian Noise)
# np.random.normal(mean, std, size) 함수 사용 
noise_r = np.random.normal(0, sigma_r, n_steps)
noise_theta = np.random.normal(0, sigma_theta_rad, n_steps)

# 4. 최종 측정값 생성 (Measurement = True Value + Noise)
z_r = true_r + noise_r
z_theta = true_theta + noise_theta

# 5. 결과 시각화 (Visualization)
plt.figure(figsize=(12, 8))

# 거리 측정 (Range) 비교
plt.subplot(2, 1, 1)
plt.plot(time, true_r, 'k-', linewidth=2, label='True Range')
plt.plot(time, z_r, 'g.', markersize=4, alpha=0.6, label='Noisy Measurement')
plt.title('Sensor Measurement: Range (r)')
plt.xlabel('Time(s)')
plt.ylabel('Range (m)')
plt.legend()
plt.grid(True)

# 방위각 측정 (Bearing) 비교
plt.subplot(2, 1, 2)
# 방위각을 도(degree) 단위로 변환하여 시각화
plt.plot(time, np.rad2deg(true_theta), 'k-', linewidth=2, label='True Theta')
plt.plot(time, np.rad2deg(z_theta), 'm.', markersize=4, alpha=0.6, label='Noisy Measurement')
plt.title('Sensor Measurement: Bearing (theta)')
plt.xlabel('Time(s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


"""
-------------------------------------------------------------------------------
Problem: Practice 2-1: EKF Implementation with Constant Velocity (CV) Model
Based on: "Optimal State Estimation" by Dan Simon & Practice_06 PDF
-------------------------------------------------------------------------------

1. 목표 (Objective):
   - 등속도(CV) 모델을 사용하는 확장 칼만 필터(EKF)를 구현하여 차량의 상태를 추정함.
   - Practice 1-2에서 생성한 측정값(Range, Bearing)을 입력으로 사용.
   - 시스템 노이즈 공분산(Q)와 측정 노이즈 공분산(R)을 튜닝하여 최적의 추정 성능 도출.

2. 시스템 모델 (System Model - Constant Velocity) [Source 3]:
   - 상태 변수 (State Vector): x = [pos_x, pos_y, vel_x, vel_y]^T (4차원)
   - 상태 전이 행렬 (F): 선형 모델
     F = [[1, 0, dT, 0],
          [0, 1, 0, dT],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
   - 프로세스 노이즈 (Q): 튜닝 파라미터 (Process Noise Covariance)

3. 측정 모델 (Measurement Model) [Source 2]:
   - 비선형 모델이므로 EKF 사용 필요.
   - 측정 함수 h(x):
     r     = sqrt(pos_x^2 + pos_y^2)
     theta = atan2(pos_y, pos_x)
   - 선형화 (Linearization): 매 시간 스텝마다 자코비안(Jacobian) 행렬 H 계산 필요.
   - 측정 노이즈 (R): 튜닝 파라미터 (Measurement Noise Covariance)

4. 초기 조건 (Initial Conditions) [Source 4]:
   - 초기 추정값 (x_hat): [0, 0, 0, 0]^T (완전한 무지 상태 가정)
   - 초기 오차 공분산 (P): 대각 행렬, 초기 불확실성을 매우 크게 설정
     P = diag(10000)

5. 구현 단계 (Steps):
   (1) Prediction: 등속도 모델(F)을 사용하여 다음 상태 예측 (x_pred, P_pred)
   (2) Jacobian Calculation: 현재 예측 상태 기준 H 행렬 계산
   (3) Update: 측정값(z)과 예측값(h(x))의 차이(Innovation)를 이용해 보정
-------------------------------------------------------------------------------
"""
# 상태 벡터 : x = [pos_x, pos_y, vel_x, vel_y]^T
n_states = 4  # 상태 변수 수
x_est = np.zeros((n_steps, n_states))  # 추정된 상태 벡터 저장
P_history = np.zeros((n_steps, n_states, n_states))  # Store covariance history

# 1. 초기 조건 설정
# 초기 추정값: 
x_est[0] = np.array([0, 0, 3.0, 3.0]) # 초기 추정값: [0, 0, 0, 0]^T

# 초기 오차 공분산 P: diag(10000)
P = np.eye(n_states) * 10000
P_history[0] = P

# 2. 시스템 모델 (Constant Velocity Model)
# F: State Transition Matrix
F = np.array([[1, 0, dT, 0],
              [0, 1, 0, dT],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])


# Q: Process Noise Covariance (Tuning Parameter)
# 실제 움직임(가속/회전)과 모델(등속 직선)의 괴리를 보상하기 위해 Q를 적절히 설정

# Option 1: Simple diagonal Q
# q_var = 0.5
# Q = np.eye(n_states) * q_var

# Option 2: Continuous white noise acceleration model (recommended)
q_acc = 1.0  # Tuning parameter
Q = np.array([
    [dT**4/4, 0,       dT**3/2, 0      ],
    [0,       dT**4/4, 0,       dT**3/2],
    [dT**3/2, 0,       dT**2,   0      ],
    [0,       dT**3/2, 0,       dT**2  ]
]) * q_acc**2

# 2.3 측정 모델 설정 (R)
# R: Measurement Noise Covariance 
R = np.array([[sigma_r**2, 0],
              [0, sigma_theta_rad**2]])

# 3. EKF 알고리즘 구현
for k in range(n_steps - 1):
   # A: Prediction Step
   # 상태 예측
   x_k = x_est[k]
   x_pred = F @ x_k  # Predicted state

   # 오차 공분산 예측: P_pred = F * P * F^T + Q
   P_pred = F @ P @ F.T + Q

   # B: Jacobian Calculation (Linearization)
   px = x_pred[0]
   py = x_pred[1]
   dist = np.sqrt(px**2 + py**2)

   # H Matrix (2x4) 계산: dh/dx
   # h(x) = [sqrt(px^2 + py^2), atan2(py, px)]
   if dist < 1e-4:  # Avoid division by zero
      dist = 1e-4
   dist2 = dist**2

   H = np.zeros((2, n_states))
   H[0, 0] = px / dist     # dr/dpx
   H[0, 1] = py / dist     # dr/dpy
   H[1, 0] = -py / dist2   # dtheta/dpx
   H[1, 1] = px / dist2    # dtheta/dpy

   # C: Update Step (Measurement Update)
   # 칼만 이득 계산: K = P_pred * H^T * (H * P_pred * H^T + R)^-1
   S = H @ P_pred @ H.T + R  # Innovation covariance
   K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman Gain

   # 측정값 예측 및 잔차(Residual) 계산
   z_pred_val = np.array([dist, np.arctan2(py, px)])
   z_meas = np.array([z_r[k+1], z_theta[k+1]])

   y_residual = z_meas - z_pred_val

   # 각도 잔차 정규화 (Angle Normalization)
   # while y_residual[1] > np.pi:
   #    y_residual[1] -= 2 * np.pi
   # while y_residual[1] < -np.pi:
   #    y_residual[1] += 2 * np.pi
   y_residual[1] = np.arctan2(np.sin(y_residual[1]), np.cos(y_residual[1]))

   # 상태 및 공분산 업데이트
   # x_next = x_pred + K * y_residual
   x_next = x_pred + K @ y_residual

   # P_next = (I - K * H) * P_pred
   P_next = (np.eye(n_states) - K @ H) @ P_pred

   # 다음 스텝을 위해 상태 및 공분산 저장
   x_est[k + 1] = x_next
   P = P_next
   P_history[k + 1] = P


# 4. 결과 시각화 (Visualization)
plt.figure(figsize=(12, 14))

# -----------------------------
# 1. 2D 궤적 비교
# -----------------------------
ax1 = plt.subplot(3, 1, 1)
ax1.plot(x, y, 'k-', label='True Trajectory')
ax1.plot(x_est[:, 0], x_est[:, 1], 'b--', label='EKF Estimate (CV Model)')
ax1.plot(
    z_r * np.cos(z_theta),
    z_r * np.sin(z_theta),
    'g.',
    markersize=1,
    alpha=0.3,
    label='Measurements'
)

ax1.set_title('2D Trajectory: True vs Estimated')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_xlim(0, 150)
ax1.grid(True)
ax1.legend()

# -----------------------------
# 2. 위치 오차
# -----------------------------
ax2 = plt.subplot(3, 1, 2)
error_x = x - x_est[:, 0]
error_y = y - x_est[:, 1]

ax2.plot(time, error_x, label='Error X')
ax2.plot(time, error_y, label='Error Y')

ax2.set_title('Position Estimation Error')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (m)')
ax2.grid(True)
ax2.legend()

# -----------------------------
# 3. 속도 상태 (vx, vy)
# -----------------------------
ax3 = plt.subplot(3, 1, 3)
ax3.plot(time, x_est[:, 2], 'r-', linewidth=1.5, label='Estimated $v_x$')
ax3.plot(time, x_est[:, 3], 'b-', linewidth=1.5, label='Estimated $v_y$')

ax3.set_title('Estimated Velocity States')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# [Practice 2-2] RMSE Calculation
# Source [1]: Get the RMSE between estimated state and true value
# -----------------------------------------------------------------------------

# 1. 속도 참값(True Velocity) 계산
vx_true = input_v * np.cos(psi)
vy_true = input_v * np.sin(psi)

# 2. 오차(Error) 계산
# 추정값(x_est)은 [pos_x, pos_y, vel_x, vel_y]^T 형태
err_pos_x = x_est[: , 0] - x  # Position X error
err_pos_y = x_est[: , 1] - y  # Position Y error
err_vel_x = x_est[: , 2] - vx_true  # Velocity X error
err_vel_y = x_est[: , 3] - vy_true  # Velocity Y error

# 3. RMSE 계산
def calculate_rmse(errors):
   mse = np.mean(errors**2)
   return np.sqrt(mse)

rmse_pos_x = calculate_rmse(err_pos_x)
rmse_pos_y = calculate_rmse(err_pos_y)
rmse_vel_x = calculate_rmse(err_vel_x)
rmse_vel_y = calculate_rmse(err_vel_y)

# 전체 위치 오차 (Distance Error) RMSE
dist_error = np.sqrt(err_pos_x**2 + err_pos_y**2)
rmse_position_total = calculate_rmse(dist_error)

# 4. 결과 출력
print("="*40)
print(f"Filter Performance (RMSE) Analysis")
print("="*40)
print(f"Position X RMSE : {rmse_pos_x:.4f} m")
print(f"Position Y RMSE : {rmse_pos_y:.4f} m")
print(f"Velocity X RMSE : {rmse_vel_x:.4f} m/s")
print(f"Velocity Y RMSE : {rmse_vel_y:.4f} m/s")
print("-" * 40)
print(f"Total Position RMSE : {rmse_position_total:.4f} m")
print("="*40)

# 5. (선택사항) 오차 그래프 시각화 보강
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, err_pos_x, label='Error X')
plt.plot(time, err_pos_y, label='Error Y')
plt.title('Position Estimation Errors')
plt.ylabel('Meters')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, err_vel_x, label='Error Vx')
plt.plot(time, err_vel_y, label='Error Vy')
plt.title('Velocity Estimation Errors')
plt.ylabel('m/s')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""
-------------------------------------------------------------------------------
Problem: Practice 2-3: P Matrix Visualization for Three Cases
Based on: Practice_06 PDF requirement
-------------------------------------------------------------------------------

Requirement: Plot the P in three cases (x-axis is time and y-axes are P_00~P_33)

Three cases mean three different tuning scenarios:
- Case 1: Low process noise (trust model more)
- Case 2: Medium process noise (balanced)
- Case 3: High process noise (trust measurements more)

NOTE: This code assumes that the following variables are already defined
from Practice 1-1, 1-2, and 2-1:
- dT, time, n_steps
- x, y, psi (true states)
- z_r, z_theta (measurements)
- sigma_r, sigma_theta_rad
-------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# FUNCTION: EKF with configurable Q, R
# ============================================================================

def run_ekf(q_acc, case_name):
    """
    Run EKF with specified process noise parameter
    
    Parameters:
    -----------
    q_acc : float
        Process noise acceleration parameter
    case_name : str
        Name of the case for plotting
        
    Returns:
    --------
    P_history : ndarray (n_steps, n_states)
        History of diagonal covariance elements
    """
    n_states = 4
    x_est = np.zeros((n_steps, n_states))
    P_history = np.zeros((n_steps, n_states))
    
    # Initial conditions
    x_est[0] = np.array([0, 0, 3.0, 3.0])
    P = np.diag([1000, 1000, 100, 100])
    P_history[0] = np.diag(P)
    
    # State transition matrix
    F = np.array([[1, 0, dT, 0],
                  [0, 1, 0, dT],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    # Process noise covariance Q
    Q = np.array([
        [dT**4/4, 0,       dT**3/2, 0      ],
        [0,       dT**4/4, 0,       dT**3/2],
        [dT**3/2, 0,       dT**2,   0      ],
        [0,       dT**3/2, 0,       dT**2  ]
    ]) * q_acc**2
    
    # Measurement noise covariance R
    R = np.array([[sigma_r**2, 0],
                  [0, sigma_theta_rad**2]])
    
    # EKF loop
    for k in range(n_steps - 1):
        # A. Prediction
        x_k = x_est[k]
        x_pred = F @ x_k
        P_pred = F @ P @ F.T + Q
        
        # B. Jacobian Calculation
        px = x_pred[0]
        py = x_pred[1]
        dist = np.sqrt(px**2 + py**2)
        
        if dist < 1e-4:
            dist = 1e-4
        dist2 = dist**2
        
        H = np.array([
            [px / dist,   py / dist,   0, 0],
            [-py / dist2, px / dist2,  0, 0]
        ])
        
        # C. Update
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        z_pred_val = np.array([dist, np.arctan2(py, px)])
        z_meas = np.array([z_r[k+1], z_theta[k+1]])
        
        y_residual = z_meas - z_pred_val
        y_residual[1] = np.arctan2(np.sin(y_residual[1]), np.cos(y_residual[1]))
        
        x_next = x_pred + K @ y_residual
        P_next = (np.eye(n_states) - K @ H) @ P_pred
        
        # Store
        x_est[k+1] = x_next
        P = P_next
        P_history[k+1] = np.diag(P)
    
    return P_history

# ============================================================================
# RUN THREE CASES (using already defined variables)
# ============================================================================

print("Running EKF for three different cases...")
P_case1 = run_ekf(q_acc=0.1, case_name="Case 1: Low Q")
P_case2 = run_ekf(q_acc=0.5, case_name="Case 2: Medium Q")
P_case3 = run_ekf(q_acc=2.0, case_name="Case 3: High Q")
print("Done!")

# ============================================================================
# VISUALIZATION 1: Three Cases Comparison
# ============================================================================

plt.figure(figsize=(14, 10))

# Define colors for three cases
colors = ['blue', 'green', 'red']
labels_cases = ['Case 1 (q=0.1)', 'Case 2 (q=0.5)', 'Case 3 (q=2.0)']
P_cases = [P_case1, P_case2, P_case3]

# Subplot 1: P_00 (Position X variance)
plt.subplot(2, 2, 1)
for i, (P_hist, label, color) in enumerate(zip(P_cases, labels_cases, colors)):
    plt.plot(time, P_hist[:, 0], color=color, linewidth=2, label=label)
plt.title('Covariance $P_{00}$ (Position X Variance)', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('Variance ($m^2$)')
plt.grid(True)
plt.legend()
plt.yscale('log')

# Subplot 2: P_11 (Position Y variance)
plt.subplot(2, 2, 2)
for i, (P_hist, label, color) in enumerate(zip(P_cases, labels_cases, colors)):
    plt.plot(time, P_hist[:, 1], color=color, linewidth=2, label=label)
plt.title('Covariance $P_{11}$ (Position Y Variance)', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('Variance ($m^2$)')
plt.grid(True)
plt.legend()
plt.yscale('log')

# Subplot 3: P_22 (Velocity X variance)
plt.subplot(2, 2, 3)
for i, (P_hist, label, color) in enumerate(zip(P_cases, labels_cases, colors)):
    plt.plot(time, P_hist[:, 2], color=color, linewidth=2, label=label)
plt.title('Covariance $P_{22}$ (Velocity X Variance)', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('Variance ($(m/s)^2$)')
plt.grid(True)
plt.legend()
plt.yscale('log')

# Subplot 4: P_33 (Velocity Y variance)
plt.subplot(2, 2, 4)
for i, (P_hist, label, color) in enumerate(zip(P_cases, labels_cases, colors)):
    plt.plot(time, P_hist[:, 3], color=color, linewidth=2, label=label)
plt.title('Covariance $P_{33}$ (Velocity Y Variance)', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('Variance ($(m/s)^2$)')
plt.grid(True)
plt.legend()
plt.yscale('log')

plt.suptitle('Practice 2-3: Covariance Matrix P Comparison (Three Cases)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 2: Single Case Detail (Optional)
# ============================================================================

plt.figure(figsize=(12, 8))

# Use Case 2 (medium) as representative
P_history = P_case2

# Subplot 1: P_00
plt.subplot(2, 2, 1)
plt.plot(time, P_history[:, 0], 'r-', linewidth=2, label='$P_{00}$')
plt.title('Covariance $P_{00}$ (Position X Variance)')
plt.xlabel('Time (s)')
plt.ylabel('Variance ($m^2$)')
plt.grid(True)
plt.legend()
plt.yscale('log')

# Subplot 2: P_11
plt.subplot(2, 2, 2)
plt.plot(time, P_history[:, 1], 'b-', linewidth=2, label='$P_{11}$')
plt.title('Covariance $P_{11}$ (Position Y Variance)')
plt.xlabel('Time (s)')
plt.ylabel('Variance ($m^2$)')
plt.grid(True)
plt.legend()
plt.yscale('log')

# Subplot 3: P_22
plt.subplot(2, 2, 3)
plt.plot(time, P_history[:, 2], 'g-', linewidth=2, label='$P_{22}$')
plt.title('Covariance $P_{22}$ (Velocity X Variance)')
plt.xlabel('Time (s)')
plt.ylabel('Variance ($(m/s)^2$)')
plt.grid(True)
plt.legend()
plt.yscale('log')

# Subplot 4: P_33
plt.subplot(2, 2, 4)
plt.plot(time, P_history[:, 3], 'm-', linewidth=2, label='$P_{33}$')
plt.title('Covariance $P_{33}$ (Velocity Y Variance)')
plt.xlabel('Time (s)')
plt.ylabel('Variance ($(m/s)^2$)')
plt.grid(True)
plt.legend()
plt.yscale('log')

plt.suptitle('Practice 2-3: Diagonal Elements of P (Case 2: q_acc=0.5)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# PRINT FINAL VALUES
# ============================================================================

print("\n" + "="*70)
print("Practice 2-3: Final Covariance Values (at t=30s)")
print("="*70)

for i, (P_hist, label) in enumerate(zip(P_cases, labels_cases), 1):
    print(f"\n{label}:")
    print(f"  P_00 (Position X variance): {P_hist[-1, 0]:.6f} m²")
    print(f"  P_11 (Position Y variance): {P_hist[-1, 1]:.6f} m²")
    print(f"  P_22 (Velocity X variance): {P_hist[-1, 2]:.6f} m²/s²")
    print(f"  P_33 (Velocity Y variance): {P_hist[-1, 3]:.6f} m²/s²")

print("\n" + "="*70)
print("Interpretation:")
print("="*70)
print("- Lower Q (Case 1): Model trusted → Lower final covariance")
print("- Higher Q (Case 3): Measurements trusted → Higher final covariance")
print("- Convergence speed: All cases converge within ~5 seconds")
print("="*70)

"""
-------------------------------------------------------------------------------
Problem: Practice 3-1: Derive the Process Model for Dead Reckoning
Based on: Practice_06 PDF
-------------------------------------------------------------------------------

Dead Reckoning Model:
- State Vector: x = [pos_x, pos_y, heading]ᵀ (3D)
- Control Input: u = [V, yaw_rate]ᵀ

Nonlinear State Equation:
  x(t+1) = x(t) + V·dT·cos(ψ(t))
  y(t+1) = y(t) + V·dT·sin(ψ(t))
  ψ(t+1) = ψ(t) + dT·ψ̇

Jacobian Matrix F = ∂f/∂x:
      ⎡ 1   0   -V·dT·sin(ψ) ⎤
  F = ⎢ 0   1    V·dT·cos(ψ) ⎥
      ⎣ 0   0    1           ⎦

-------------------------------------------------------------------------------
"""

print("="*70)
print("Practice 3-1: Process Model Jacobian Derivation")
print("="*70)
print("\nDead Reckoning Nonlinear State Equation:")
print("  x(t+1) = x(t) + V·dT·cos(ψ(t))")
print("  y(t+1) = y(t) + V·dT·sin(ψ(t))")
print("  ψ(t+1) = ψ(t) + dT·ψ̇")
print("\nState Vector: x = [x, y, ψ]ᵀ (3D)")
print("Control Input: u = [V, ψ̇]ᵀ")
print("\nJacobian Matrix F = ∂f/∂x:")
print("\n    ⎡ 1   0   -V·dT·sin(ψ) ⎤")
print("F = ⎢ 0   1    V·dT·cos(ψ) ⎥")
print("    ⎣ 0   0    1           ⎦")
print("\nNote: F is evaluated at the predicted state at each time step")
print("="*70)

# Example: Calculate F at t=0
V_example = input_v[0]
psi_example = psi[0]

F_example = np.array([
    [1, 0, -V_example * dT * np.sin(psi_example)],
    [0, 1,  V_example * dT * np.cos(psi_example)],
    [0, 0,  1]
])

print(f"\nExample: F matrix at t=0 (V={V_example:.2f}, ψ={np.rad2deg(psi_example):.2f}°):")
print(F_example)
print("="*70)