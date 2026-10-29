hernando_path = 'W:\radiologie\mrt-probanden\AG_Laun\Matthias Rohe\MATLAB\fwtoolbox\hernando';
% datapath = 'W:\radiologie\mrt-probanden\AG_Laun\Matthias Rohe\Daten\2025_10_06_Cima_B0';

addpath(genpath(hernando_path));


cd 'W:\radiologie\mrt-probanden\AG_Laun\Matthias Rohe\Daten\2025_10_06_Cima_B0'
%%


clear imDataParams algoParams;

% Show start time and print status
disp(datetime('now'));
disp('Running...');


% Allocate image data parameters
imDataParams.images = images_array(:,:,:,:,:);
imDataParams.TE = meanTimes(:)/1e6; %[2.2, 4.5, 6.8, 9.1, 11.4, 13.7] /1e3;
imDataParams.FieldStrength = 3.0;
imDataParams.PrecessionIsClockwise = 1;

algoParams.USE_REF_MAPS = 0;
%imDataParams.FieldMap = outParams_images.fieldmap;
%imDataParams.R2StarMap = outParams_images.r2starmap;

algoParams.species(1).name = 'water';
algoParams.species(2).name = 'fat';
algoParams.species(1).frequency = [0]; 
algoParams.species(1).relAmps = [1];  
algoParams.species(2).frequency = [-3.926, -3.5, -3.211, -2.782, -2.563, -2.037, -0.724, -0.510, 0.513];
% Phantom: [-3.926, -3.507, -3.211, -2.782, -2.563, -2.037, -0.724, -0.510, 0.513]
% In Vivo: [-3.6, -3.2, -2.91, -2.47, -2.25, -1.73, 0.81]
algoParams.species(2).relAmps = [0.0879, 0.6478, 0.0130, 0.0936, 0.0468, 0.0031, 0.0230, 0.0158, 0.0690];
% Phantom: [0.0879, 0.6478, 0.0130, 0.0936, 0.0468, 0.0031, 0.0230, 0.0158, 0.0690]
% In Vivo: [0.0813, 0.6006, 0.0678, 0.0914, 0.0635, 0.0152, 0.0802]
% In Vivo Resonance Areas:[128, 945.7, 106.8, 143.9, 100, 24, 126.3]  total: 1574.7

% Allocate algorithm parameters
algoParams.size_clique = 1; % Size of MRF neighborhood (1 uses an 8-neighborhood, common in 2D)
algoParams.range_r2star = [0 0]; % Range of R2* values
algoParams.NUM_R2STARS = 1; % Number of R2* values for quantization
algoParams.range_fm = [-400 400]; % Range of field map values
algoParams.NUM_FMS = 301; % Number of field map values to discretize
algoParams.NUM_ITERS = 40; % Number of graph cut iterations
algoParams.SUBSAMPLE = 2; % Spatial subsampling for field map estimation (for speed)
algoParams.DO_OT = 1; % 0,1 flag to enable optimization transfer descent (final stage of field map estimation)
algoParams.LMAP_POWER = 2; % Spatially-varying regularization (2 gives ~ uniformn resolution)
algoParams.lambda = 0.01; % Regularization parameter
algoParams.LMAP_EXTRA = 0.0; % More smoothing for low-signal regions
algoParams.TRY_PERIODIC_RESIDUAL = 0; % Take advantage of periodic residual if uniform TEs (will change range_fm) 


% Calculate output

outParams = fw_i2cm1i_3pluspoint_hernando_graphcut(imDataParams, algoParams);


outParams.fatfraction = abs(outParams.species(2).amps) ./ (abs(outParams.species(2).amps) + abs(outParams.species(1).amps));


% Show end time
disp(datetime('now'));

%%

%fatfraction = Compute_FatFraction(outParams, outParams_phase_shift);
