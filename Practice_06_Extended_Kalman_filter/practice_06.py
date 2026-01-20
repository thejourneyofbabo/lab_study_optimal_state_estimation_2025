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