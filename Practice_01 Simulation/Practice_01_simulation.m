%%  File Name: 01_Simulation %%%%%%%%%%%%%%%%%%%%
%
%  Copyright 2020 AI Lab. Konkuk Univ. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;


%% [Simulation configuration]

delta_t = 0.1;   		% [s]
start_t = 0.;    		% [s]
end_t = 10.;    		% [s]
sim_time = start_t : delta_t : end_t;
sim_total_steps = length( sim_time );


%% [Practice 1]
% Please simulate the r_t, v_t, and a_t from time t=0 to t=10. 
% Plot the simulation (x-axis is time and y-axes are r_t, v_t, and a_t.

% Simulation variables

sim_acceleration = zeros(sim_total_steps,1);
sim_velocity = zeros(sim_total_steps ,1);
sim_altitude = zeros(sim_total_steps ,1);

init_acceleration = -9.80665;   	% [m/s^2]
init_velocity = 40.;   		% [m/s]
init_altitude = 100.;   		% [m]

% Generate simulation data
isFirstStep = true;
for idxSim = 1 : sim_total_steps 
    
    if isFirstStep  == true
        
       %%%% TODO %%%%
       sim_acceleration( idxSim ) = init_acceleration;
       sim_velocity( idxSim ) = init_velocity;
       sim_altitude( idxSim ) = init_altitude;
       isFirstStep = false;
       
       continue;
    end
    
    %%%% TODO %%%%
    sim_acceleration( idxSim ) = sim_acceleration(idxSim - 1) ;
    sim_velocity( idxSim ) = sim_velocity(idxSim -1) + sim_acceleration(idxSim -1)*delta_t ;
    sim_altitude( idxSim ) = sim_altitude(idxSim -1 ) + sim_velocity(idxSim - 1) * delta_t * + (1/2)*sim_acceleration(idxSim -1)*delta_t*delta_t ;
    
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

%% [Practice 2]
% Please simulate r_t^range, which is a range sensor simulation data for the previous simulation. 
% The standard deviation of the sensor noise is 0.1m. Plot the simulation results and save to m file. 

% Simulation variables
altitude_sensor_std = 0.1; % [m]
sim_altitude_sensor = zeros(sim_total_steps ,1);

% Simulate the noisy measurement
for idxSim = 1 : sim_total_steps 

    %%%% TODO %%%%
    % sim_altitude_sensor(idxSim) = XXXXX ;

end


% Plot
figure('Name','Practice 2','Position',[100,100,840,630]);
hold on;
plot(sim_time, sim_altitude, 'r.');
plot(sim_time, sim_altitude_sensor, 'b.');
legend('true altitude', 'measured altitude'); xlabel('time[s]'); ylabel('altitude [m]'); grid on;
hold off;