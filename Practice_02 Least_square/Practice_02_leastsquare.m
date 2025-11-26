%%  File Name: 01_Simulation %%%%%%%%%%%%%%%%%%%%
%
%  Copyright 2020 AI Lab. Konkuk Univ. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;

%% [Load data from mat file]
m = matfile('data_altitude_sensor.mat');

sim_time = m.sim_time;                       % [s]
sim_total_steps = m.sim_total_steps;

sim_true = m.sim_true;                       % r0, v0, a0;
sim_altitude = m.sim_altitude;               % [m]
sim_altitude_sensor = m.sim_altitude_sensor; % [m]

%% [Practice 1]
% Please find the r0, v0, and a0 using given data through least square estimation.

Num_of_state = 3; % r0, v0, a0
H = zeros(sim_total_steps ,Num_of_state);

for idxSim = 1: sim_total_steps
    % ToDo
    % H(idxSim,:) = XXXXXX ;

end

% Calculate estimated value
% ToDo
% x_ls = XXXXXX ;    
error_ls = abs(x_ls-sim_true);

fprintf('[Practice 1] \n');
fprintf('r0: %f, v0: %f, a0: %f   \n', x_ls(1), x_ls(2), x_ls(3));
fprintf('r0_error: %f, v0_error: %f, a0_error: %f   \n', error_ls(1), error_ls(2), error_ls(3));



%% [Practice 2]
% Please find the r0, v0, and a0 using given data through recursive least square estimation.

Num_of_state = 3; % r0, v0, a0
H = zeros(sim_total_steps ,Num_of_state);
x_rls = zeros(Num_of_state, sim_total_steps);
P_rls = cell(sim_total_steps,1);

% Initialize
P0 = [100  0   0;
       0  100  0;
       0   0  100];
x0 = [0 0 0]';
% ToDo
% R = XXX ;

isFirstStep = true;
for idxSim = 1: sim_total_steps
    
    H(idxSim,:) = [1 sim_time(idxSim) 1/2*sim_time(idxSim)*sim_time(idxSim)];
    
    if isFirstStep  == true
        
        x_rls(:,idxSim) = x0;
        P_rls{idxSim} = P0;
        isFirstStep = false;
       
        continue;
    end
    
    % ToDo
    % K = XXXXXX ;
    % x_rls(:,idxSim) = XXXXXX ;
    % P_rls{idxSim} = XXXXXX ;
           
end

% Plot
error_rls = abs(x_rls-sim_true);

figure('Name','[Practice 2]','Position',[100,100,1200,630]);
subplot(3,2,1)
hold on;
plot(sim_time, x_rls(1,:), 'r.-');
plot(sim_time, sim_true(1)*ones(sim_total_steps,1),'k');
legend('estimated r_{0}','true r_{0}'); 
xlabel('time[s]'); ylabel('r_{0} [m]'); grid on; hold off;

subplot(3,2,2)
hold on;
plot(sim_time, error_rls(1,:), 'k.-');
legend('error of r_{0}'); 
xlabel('time[s]'); ylabel('error of r_{0} [m]'); grid on; hold off;

subplot(3,2,3)
hold on;
plot(sim_time, x_rls(2,:), 'b.-');
plot(sim_time, sim_true(2)*ones(sim_total_steps,1),'k');
legend('estimated v_{0}','true v_{0}');
xlabel('time[s]'); ylabel('v_{0} [m/s]'); grid on; hold off;

subplot(3,2,4)
hold on;
plot(sim_time, error_rls(2,:), 'k.-');
legend('error of v_{0}');
xlabel('time[s]'); ylabel('error of v_{0} [m/s]'); grid on; hold off;

subplot(3,2,5)
hold on;
plot(sim_time, x_rls(3,:), 'g.-');
plot(sim_time, sim_true(3)*ones(sim_total_steps,1),'k');
legend('estimated a_{0}', 'true a_{0}');
xlabel('time[s]'); ylabel('a_{0} [m/s]'); grid on; hold off;

subplot(3,2,6)
hold on;
plot(sim_time, error_rls(3,:), 'k.-');
legend('error of a_{0}');
xlabel('time[s]'); ylabel('error of a_{0} [m/s]'); grid on; hold off;
