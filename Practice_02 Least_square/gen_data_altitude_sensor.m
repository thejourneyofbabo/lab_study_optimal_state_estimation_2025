clear;
clc;
close all;


%% Q1. Generate t,r,v,a simulation sata
% Initialize
delta_t = 0.1;   % [s]
start_t = 0.;    % [s]
end_t = 5.;      % [s]
sim_time = start_t : delta_t : end_t;
sim_total_steps = length( sim_time);

sim_acceleration = zeros(sim_total_steps,1);
sim_velocity = zeros(sim_total_steps,1);
sim_altitude = zeros(sim_total_steps,1);

init_acceleration = -9.80665;   	% [m/s^2]
init_velocity = -10.;   		% [m/s]
init_altitude = 180.;   		% [m]

true_x = [init_altitude init_velocity init_acceleration]';
% Generate simulation data
isFirstStep = true;
for idxSim = 1:sim_total_steps
    if isFirstStep == true
        sim_acceleration( idxSim ) = init_acceleration;
        sim_velocity( idxSim ) = init_velocity;
        sim_altitude( idxSim ) = init_altitude;
        isFirstStep = false;
        continue;
    end
    
    sim_acceleration( idxSim ) = init_acceleration;
    sim_velocity( idxSim ) = sim_velocity( idxSim -1 ) + sim_acceleration( idxSim -1 ) * delta_t;
    sim_altitude( idxSim ) = sim_altitude( idxSim -1 ) + sim_velocity( idxSim -1 ) * delta_t + 0.5 * sim_acceleration(idxSim-1) * delta_t^2;
end

% Plot
figure('Name','Practice 1','Position',[100,100,840,630]);

hold on; 

subplot(2,1,1)
plot(sim_time, sim_altitude, 'r');
legend('altitude'); xlabel('time[s]'); ylabel('altitude[m]'); grid on;

subplot(2,1,2)
plot(sim_time, sim_velocity, 'b');
legend('velocity');
xlabel('time[s]'); ylabel('velocity [m/s]'); grid on;

hold off


%% Q2. Generate r_noise simulation data
% Initialize
altitude_sensor_std = 0.1; % [m]
sim_altitude_sensor = zeros(sim_total_steps ,1);
sim_noise = zeros(sim_total_steps,1);

% Simulate the noisy measurement
for idxSim = 1 : sim_total_steps 
    
    noise = normrnd(0, altitude_sensor_std);
    
    sim_altitude_sensor(idxSim) = sim_altitude(idxSim) + noise;
    
    sim_noise(idxSim) = noise; 
    
end

% Plot
figure('Name','Practice 2','Position',[100,100,840,630]);
hold on;
plot(sim_time, sim_altitude, 'r.');
plot(sim_time, sim_altitude_sensor, 'b.');
legend('true altitude', 'measured altitude'); xlabel('time[s]'); ylabel('altitude [m]'); grid on;
hold off;


mat = matfile('data_altitude_sensor.mat', 'Writable', true);
mat.sim_time = sim_time;
mat.sim_total_steps = sim_total_steps;

mat.sim_true = true_x;
mat.sim_velocity = sim_velocity;
mat.sim_altitude = sim_altitude;
mat.sim_acceleration = sim_acceleration;
mat.sim_altitude_sensor = sim_altitude_sensor;
mat.sim_noise = sim_noise;