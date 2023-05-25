NUMBER_OF_SENSORS = 7;
ULA = phased.ULA('NumElements', NUMBER_OF_SENSORS, 'ElementSpacing',0.01);
ULA.Element.FrequencyRange = [0 40e3];
UCA = phased.UCA('NumElements', NUMBER_OF_SENSORS, 'Radius', 1.30656);

% sensorArrayAnalyzer;

OPERATING_FREQUENCY = 33e3;
PHASE_VELOCITY = 770;

NUMBER_OF_SIGNALS = 4;

FILE_NAME_ULA = "comsol_simulations_analytic_signals_ULA.csv";
signal_ula= readtable(FILE_NAME_ULA);
signal_ula = table2array(signal_ula);
% FILE_NAME_UCA = "comsol_simulations_analytic_signals_UCA.csv";
% signal_uca = readtable(FILE_NAME_UCA);
% signal_uca = table2array(signal_uca);
% The dumb way of bypassing the other geometry:
signal_uca = signal_ula;

%%

rootmusicangle_ula = phased.RootMUSICEstimator('SensorArray',ULA,...
    'OperatingFrequency',OPERATING_FREQUENCY,...
    'NumSignalsSource','Property','NumSignals', NUMBER_OF_SIGNALS, ...
    'PropagationSpeed', PHASE_VELOCITY);

rootmusicangle_ula.ForwardBackwardAveraging = true;
rootmusicangle_ula.SpatialSmoothing = 0;

root_music_ula_ang = rootmusicangle_ula(signal_ula);
root_music_sorted_ula_angles = sort(root_music_ula_ang);
TEST_PARAMETERS = false;
if TEST_PARAMETERS
    error_min = Inf; % Initialize the minimum error to a large value
    phase_velocity_min = 0; % Initialize the corresponding phase_velocity
    
    target_angles = [-143.973, -41.634, 33.690, 151.390];
    
    for phase_velocity = 400:1:650
        release(rootmusicangle_ula)
        rootmusicangle_uca = phased.RootMUSICEstimator('SensorArray',UCA,...
            'OperatingFrequency',OPERATING_FREQUENCY,...
            'NumSignalsSource','Property','NumSignals', NUMBER_OF_SIGNALS, ...
            'PropagationSpeed', phase_velocity);
        rootmusicangle_uca.ForwardBackwardAveraging = true;
        rootmusicangle_uca.SpatialSmoothing = 1;
        ELEVATION_ANGLE = 0;
        root_music_uca_ang = rootmusicangle_uca(signal_uca, ELEVATION_ANGLE);
        root_music_uca_ang = rotate_angles(root_music_uca_ang, NUMBER_OF_SENSORS);
        root_music_sorted_uca_angles = sort(root_music_uca_ang);
        
        % Compute the error between root_music_sorted_uca_angles and target_angles
        error = sum(abs(root_music_sorted_uca_angles - target_angles));
        
        % Check if the current error is lower than the minimum error
        if error < error_min
            error_min = error;
            phase_velocity_min = phase_velocity;
        end
    end
    disp('Minimum Error:')
    disp(error_min)
    disp('Phase Velocity yielding the lowest error:')
    disp(phase_velocity_min)
    PHASE_VELOCITY = phase_velocity_min
end

%%

rootmusicangle_uca = phased.RootMUSICEstimator('SensorArray',UCA,...
    'OperatingFrequency',OPERATING_FREQUENCY,...
    'NumSignalsSource','Property','NumSignals', NUMBER_OF_SIGNALS, ...
    'PropagationSpeed', PHASE_VELOCITY);
rootmusicangle_uca.ForwardBackwardAveraging = true;
rootmusicangle_uca.SpatialSmoothing = 1;
ELEVATION_ANGLE = 0;
root_music_uca_ang = rootmusicangle_uca(signal_uca, ELEVATION_ANGLE);
root_music_uca_ang = rotate_angles(root_music_uca_ang, NUMBER_OF_SENSORS);
root_music_sorted_uca_angles = sort(root_music_uca_ang);

%%


musicangle = phased.MUSICEstimator('SensorArray',ULA,...
    'OperatingFrequency',OPERATING_FREQUENCY,'ForwardBackwardAveraging',true,...
    'NumSignalsSource','Property','NumSignals', NUMBER_OF_SIGNALS,...
    'DOAOutputPort',true, ...
    'PropagationSpeed', PHASE_VELOCITY);

[~,music_ang] = musicangle(signal_ula);
music_sorted_angles = sort(music_ang);
plotSpectrum(musicangle)

%%

rootwsfangle = phased.RootWSFEstimator('SensorArray',ULA,...
    'OperatingFrequency',OPERATING_FREQUENCY,'MaximumIterationCount',50, ...
    'PropagationSpeed', PHASE_VELOCITY, 'NumSignalsSource', 'Property', ...
    'NumSignals', NUMBER_OF_SIGNALS);
wsf_ang = rootwsfangle(signal_ula);
wsf_sorted_angles = sort(wsf_ang);

%%

esprit = phased.ESPRITEstimator('SensorArray',ULA,...
    'OperatingFrequency',OPERATING_FREQUENCY,'ForwardBackwardAveraging',true,...
    'PropagationSpeed', PHASE_VELOCITY, ...
    'NumSignalsSource','Property', ...
    'NumSignalsMethod','AIC', ...
    'NumSignals', NUMBER_OF_SIGNALS);
esprit_angles = esprit(signal_ula);
esprit_sorted_angles = sort(esprit_angles);

%%

METHOD_NAMES_ULA = ["Root-WSF", "Root-MUSIC", "MUSIC", "ESPRIT"];
method_results_ula = [wsf_sorted_angles; root_music_sorted_ula_angles; music_sorted_angles; esprit_sorted_angles];
% Transpose the matrix
method_results_ula_transposed = method_results_ula.';

% Combine the headers and data into a single cell array
data_ula = [METHOD_NAMES_ULA; num2cell(method_results_ula_transposed)];

% Write the data_ula to a CSV file using the built-in `writematrix` function
filename = strcat('results_angles_estimation_ULA.csv');
writematrix(data_ula, filename);

METHOD_NAMES_UCA = ["Root-MUSIC"];
method_results_uca = [root_music_sorted_uca_angles];
% Transpose the matrix
method_results_uca_transposed = method_results_uca.';

% Combine the headers and data into a single cell array
data_uca = [METHOD_NAMES_UCA; num2cell(method_results_uca_transposed)];

% Write the data_uca to a CSV file using the built-in `writematrix` function
filename = strcat('results_angles_estimation_UCA.csv');
writematrix(data_uca, filename);

%%

function rotated_angles = rotate_angles(angle_list, number_of_sensors)
rotation_amount = ((number_of_sensors - 1) / 2) * (360 / number_of_sensors);
rotated_angles = mod(angle_list + rotation_amount + 180, 360) - 180;
end
