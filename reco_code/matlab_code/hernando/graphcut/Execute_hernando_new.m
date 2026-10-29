%[algoParams, outParams] = Execute_hernando_new(finalImages_array, meanTimes, "invivo", 7, 0)
%meanTimes = [[2.2e3 4.5e3 6.8e3 9.1e3 11.4e3 13.7e3]]

function [algoParams, outParams] = Execute_hernando_new(file_name, file_path, meanTimes, model, field_strength, phase_shift, outParams_images)

load([file_path file_name '.mat'], 'finalImages_array');
images_array = finalImages_array;
addpath('../common/')
addpath('../matlab_bgl/')
addpath('../create_synthetic/')
addpath('../descent/')
addpath('../graphcut/')
addpath('../matlab_bgl-3.0-beta/')
addpath('../mixed_fitting/')
addpath('../')
% Show start time

mainTimer = tic;
disp('Busy - FWS');

% Allocate image data parameters
imDataParams.images = images_array;
imDataParams.TE = meanTimes(:)/1e6; %meanTimes(:)/1e6; [2.2, 4.5, 6.8, 9.1, 11.4, 13.7] /1e3;  [2.2, 5.4, 8.6, 11.8, 15, 18.2] /1e3;
imDataParams.FieldStrength = field_strength;
imDataParams.PrecessionIsClockwise = 1;

algoParams.USE_REF_MAPS = 0;
if algoParams.USE_REF_MAPS == 1
    imDataParams.FieldMap = outParams_images.fieldmap(:,:,1);
    imDataParams.R2StarMap = outParams_images.r2starmap(:,:,1);
end

algoParams.species(1).name = 'water';
algoParams.species(2).name = 'fat';
algoParams.species(1).frequency = [0] + phase_shift; 
algoParams.species(1).relAmps = [1];  

if model == "invivo"
    algoParams.species(2).frequency = [-3.6, -3.2, -2.91, -2.47, -2.25, -1.73, 0.81] + phase_shift;
    algoParams.species(2).relAmps = [0.0813, 0.6006, 0.0678, 0.0914, 0.0635, 0.0152, 0.0802];
elseif model == "phantom"
    algoParams.species(2).frequency = [-3.926, -3.5, -3.211, -2.782, -2.563, -2.037, -0.724, -0.510, 0.513] + phase_shift;
    algoParams.species(2).relAmps = [0.0879, 0.6478, 0.0130, 0.0936, 0.0468, 0.0031, 0.0230, 0.0158, 0.0690];
elseif model == "phantom_numerical"
    algoParams.species(2).frequency = [-3.659, -3.236, -2.905, -2.486, -2.27, -1.709, -0.393, 0.781] + phase_shift;
    algoParams.species(2).relAmps = [0.082, 0.623, 0.015, 0.096, 0.048, 0.009, 0.026, 0.018];
elseif model == "dixRAVE"
    algoParams.species(2).frequency = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60] + phase_shift;
    algoParams.species(2).relAmps = [0.087 0.693 0.128 0.004 0.039 0.048];
end

% Phantom: [-3.926, -3.507, -3.211, -2.782, -2.563, -2.037, -0.724, -0.510, 0.513]
% In Vivo: [-3.6, -3.2, -2.91, -2.47, -2.25, -1.73, 0.81]
% Phantom: [0.0879, 0.6478, 0.0130, 0.0936, 0.0468, 0.0031, 0.0230, 0.0158, 0.0690]
% In Vivo: [0.0813, 0.6006, 0.0678, 0.0914, 0.0635, 0.0152, 0.0802]
% In Vivo Resonance Areas:[128, 945.7, 106.8, 143.9, 100, 24, 126.3]  total: 1574.7

% Allocate algorithm parameters
algoParams.size_clique = 1; % Size of MRF neighborhood (1 uses an 8-neighborhood, common in 2D)
algoParams.range_r2star = [0 400]; % Range of R2* values
algoParams.NUM_R2STARS = 101; % Number of R2* values for quantization
algoParams.range_fm = [-400 400]; % Range of field map values
algoParams.NUM_FMS = 301; % Number of field map values to discretize
algoParams.NUM_ITERS = 40; % Number of graph cut iterations
algoParams.SUBSAMPLE = 2; % Spatial subsampling for field map estimation (for speed)
algoParams.DO_OT = 1; % 0,1 flag to enable optimization transfer descent (final stage of field map estimation)
algoParams.LMAP_POWER = 2; % Spatially-varying regularization (2 gives ~ uniformn resolution)
algoParams.lambda = 0.05; % Regularization parameter
algoParams.LMAP_EXTRA = 0.05; % More smoothing for low-signal regions
algoParams.TRY_PERIODIC_RESIDUAL = 0; % Take advantage of periodic residual if uniform TEs (will change range_fm) 


% Calculate output

outParams = fw_i2cm1i_3pluspoint_hernando_graphcut(imDataParams, algoParams);


outParams.fatfraction = abs(outParams.species(2).amps) ./ (abs(outParams.species(2).amps) + abs(outParams.species(1).amps));

% Show total run time
elapsedTime = toc(mainTimer);
minutes = floor(mod(elapsedTime, 3600) / 60); seconds = floor(mod(elapsedTime, 60));
fprintf('Total FWS time: %dmin %ds\n', minutes, seconds);

save([file_path 'graphcut_res\' 'outParams_' file_name '.mat'], 'outParams')


%clear elapsedTime mainTimer minutes seconds imDataParams;


%%

%fatfraction = Compute_FatFraction(outParams, outParams_phase_shift);
