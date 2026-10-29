
clear imDataParams algoParams;

% Show start time and print status
disp(datetime('now'));
disp('Running...');

% Allocate image data parameters
imDataParams.FieldStrength = 7.0;
imDataParams.PrecessionIsClockwise = 1;

algoParams.resonance = 1; % 1 for water  2 for fat

% Allocate fat/water parameters
if algoParams.resonance == 1
    algoParams.species(1).name = 'water';
    algoParams.species(2).name = 'fat';
elseif algoParams.resonance == 2
    algoParams.species(2).name = 'water';
    algoParams.species(1).name = 'fat';
end
algoParams.species(1).frequency = [0]+3.5; 
algoParams.species(1).relAmps = [1];  
algoParams.species(2).frequency = [-3.926, -3.50, -3.211, -2.782, -2.563, -2.037, -0.724, -0.510, 0.513]+3.5;
% Phantom: [-3.926, -3.507, -3.211, -2.782, -2.563, -2.037, -0.724, -0.510, 0.513]
% In Vivo: [-3.6, -3.2, -2.91, -2.47, -2.25, -1.73, 0.81]
algoParams.species(2).relAmps = [0.0879, 0.6478, 0.0130, 0.0936, 0.0468, 0.0031, 0.0230, 0.0158, 0.0690];
% Phantom: [0.0879, 0.6478, 0.0130, 0.0936, 0.0468, 0.0031, 0.0230, 0.0158, 0.0690]
% In Vivo: [0.0813, 0.6006, 0.0678, 0.0914, 0.0635, 0.0152, 0.0802]
% In Vivo Resonance Areas:[128, 945.7, 106.8, 143.9, 100, 24, 126.3]  total: 1574.7

% Allocate algorithm parameters
algoParams.size_clique = 1; % Size of MRF neighborhood (1 uses an 8-neighborhood, common in 2D)
algoParams.range_r2star = [0 200]; % Range of R2* values
algoParams.NUM_R2STARS = 51; % Number of R2* values for quantization
algoParams.range_fm = [-400 400]; % Range of field map values
algoParams.NUM_FMS = 301; % Number of field map values to discretize
algoParams.NUM_ITERS = 40; % Number of graph cut iterations
algoParams.SUBSAMPLE = 2; % Spatial subsampling for field map estimation (for speed)
algoParams.DO_OT = 1; % 0,1 flag to enable optimization transfer descent (final stage of field map estimation)
algoParams.LMAP_POWER = 2; % Spatially-varying regularization (2 gives ~ uniformn resolution)
algoParams.lambda = 0.05; % Regularization parameter
algoParams.LMAP_EXTRA = 0.05; % More smoothing for low-signal regions
algoParams.TRY_PERIODIC_RESIDUAL = 0; % Take advantage of periodic residual if uniform TEs (will change range_fm) 

water_images = zeros(size(sliced_complex_array, 1), ...
                     size(sliced_complex_array, 1), ...
                     34);
fat_images = zeros(size(sliced_complex_array, 1), ...
                     size(sliced_complex_array, 1), ...
                     34);
fatfractions = zeros(size(sliced_complex_array, 1), ...
                     size(sliced_complex_array, 1), ...
                     34);
         
k = 1;

% Calculate output
for i=10:5:length(echotimes)
    
    imDataParams.images = sliced_complex_array(:,:,1,1,1:i);
    imDataParams.TE = echotimes(1:i)/1e6;
    
    outParams = fw_i2cm1i_3pluspoint_hernando_graphcut(imDataParams, algoParams);

    if algoParams.resonance == 1
        outParams.fatfraction = abs(outParams.species(2).amps) ./ (abs(outParams.species(2).amps) + abs(outParams.species(1).amps));
    elseif algoParams.resonance == 2
        outParams.fatfraction = abs(outParams.species(1).amps) ./ (abs(outParams.species(2).amps) + abs(outParams.species(1).amps));
    end
    
    water_images(:,:,k) = outParams.species(1).amps;
    fat_images(:,:,k) = outParams.species(2).amps;
    fatfractions(:,:,k) = outParams.fatfraction;
    
    disp(datetime('now'));
    disp(['Step ' num2str(k) ' ready!'])
    k = k+1;
end

clear i k;

%%

fatfraction = zeros(size(outParams.species(1).amps));

% Loop through each pixel
for i = 1:size(outParams.species(1).amps, 1)
    for j = 1:size(outParams.species(1).amps, 2)
        if abs(outParams_phase_shift.species(2).amps(i, j)) >= abs(outParams.species(1).amps(i, j))
            % Perform a calculation with image1's pixel
            fatfraction(i, j) = abs(outParams_phase_shift.species(2).amps(i, j)) ./ (abs(outParams.species(1).amps(i, j) + outParams_phase_shift.species(2).amps(i, j)));
        else
            % Perform a calculation with image2's pixel
            fatfraction(i, j) = 1 - abs(outParams.species(1).amps(i, j)) ./ (abs(outParams.species(1).amps(i, j) + outParams_phase_shift.species(2).amps(i, j)));
        end
    end
end

clear i j;


%%
num_images = size(fatfractions, 3); % Automatically determine number of images (e.g., 38)
num_rois = 9;                       % Number of ROIs/vials
ground_truth = [0, 5, 10, 15, 20, 25, 50, 75, 100]; % Ground truth fat fractions
roi_names = {'Vial_0', 'Vial_5', 'Vial_10', 'Vial_15', 'Vial_20', 'Vial_25', 'Vial_50', 'Vial_75', 'Vial_100'}; 

% === ROI Selection on One Reference Image ===
ref_image = fatfractions(:,:,30); % Choose any representative image (e.g., 30th slice)
roi_masks = cell(1, num_rois);    % Store ROI masks

figure, imshow(ref_image, []), title('Draw 9 ROIs from 0% to 100%');
colormap('jet');
hold on;
for r = 1:num_rois
    h = drawcircle(); % Manually draw ROI
    roi_masks{r} = createMask(h); % Store binary mask
end
close;

% === Main Evaluation Loop ===
results = struct(); % Initialize results struct

for i = 1:num_images
    img = double(fatfractions(:,:,i)); % Get the i-th image slice
    abs_errors = zeros(1, num_rois);   % To store absolute errors per ROI

    results(i).Image_Index = i;        % Optional: store image index
    for r = 1:num_rois
        roi_pixels = img(roi_masks{r});         % Extract ROI pixels
        mean_intensity = mean(roi_pixels(:));   % Mean fat fraction
        diff_value = mean_intensity * 100 - ground_truth(r); % Error

        results(i).(roi_names{r}) = diff_value; % Store difference
        abs_errors(r) = abs(diff_value);        % Track absolute error
    end
    results(i).Mean_Absolute_Error = mean(abs_errors); % Overall MAE for image
end


clear diff_value ground_truth h i img mean_intensity measurements mri_images num_images num_rois r ref_image roi_names roi_pixels abs_errors;

%%
fatfractions_optimized = zeros(size(fatfractions));
% Loop through each pixel
for k = 1:size(fatfractions, 3)
    for i = 1:size(fatfractions, 1)
        for j = 1:size(fatfractions, 2)
            if abs(fat_images_phase_shift(i, j, k)) >= abs(water_images(i, j, k))
                % Perform a calculation with image1's pixel
                fatfractions_optimized(i, j, k) = abs(fat_images_phase_shift(i, j, k)) ./ (abs(water_images(i, j, k) + fat_images_phase_shift(i, j, k)));
            else
                % Perform a calculation with image2's pixel
                fatfractions_optimized(i, j, k) = 1 - abs(water_images(i, j, k)) ./ (abs(water_images(i, j, k) + fat_images_phase_shift(i, j, k)));
            end
        end
    end
end

clear i j k;

%%
fatfractions_noisebias = zeros(size(fatfractions));
% Loop through each pixel
for k = 1:size(fatfractions, 3)
	% Perform a calculation with image1's pixel
	fatfractions_noisebias(:, :, k) = abs(fat_images_phase_shift(:, :, k)) ./ (abs(water_images(:, :, k) + fat_images_phase_shift(:, :, k)));
end

clear i j k;

%%
num_images = size(fatfractions_noisebias, 3); % Automatically determine number of images (e.g., 38)
num_rois = 9;                       % Number of ROIs/vials
ground_truth = [0, 5, 10, 15, 20, 25, 50, 75, 100]; % Ground truth fat fractions
roi_names = {'Vial_0', 'Vial_5', 'Vial_10', 'Vial_15', 'Vial_20', 'Vial_25', 'Vial_50', 'Vial_75', 'Vial_100'}; 

% === Main Evaluation Loop ===
results_optimized = struct(); % Initialize results struct

for i = 1:num_images
    img = double(fatfractions_noisebias(:,:,i)); % Get the i-th image slice
    abs_errors = zeros(1, num_rois);   % To store absolute errors per ROI

    results_optimized(i).Image_Index = i;        % Optional: store image index
    for r = 1:num_rois
        roi_pixels = img(roi_masks{r});         % Extract ROI pixels
        mean_intensity = mean(roi_pixels(:));   % Mean fat fraction
        diff_value = mean_intensity * 100 - ground_truth(r); % Error

        results_optimized(i).(roi_names{r}) = diff_value; % Store difference
        abs_errors(r) = abs(diff_value);        % Track absolute error
    end
    results_optimized(i).Mean_Absolute_Error = mean(abs_errors); % Overall MAE for image
end


clear diff_value ground_truth h i img mean_intensity measurements mri_images num_images num_rois r ref_image roi_names roi_pixels abs_errors;

%%
% Extract MAE values from results struct
mae_values = [results.Mean_Absolute_Error];  % 1×num_images vector

% Plot
figure;
plot((echotimes(10:5:end)+108)/1e3, mae_values, '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Cutoff echo time [ms]');
ylabel('MAE [%]');
%ylim([2 5.5]);
set(gca, 'FontSize', 17);

exportgraphics(gcf, 'W:\radiologie\mrt-probanden\AG_Laun\Matthias Rohe\Daten\2025_07_13_NumberOfEchoes\2025_04_23_GTT\Bilder\R2_Results_Non_Optimized.png', 'Resolution', 500);


%%
% Extract MAE values from results struct
mae_values = [results_optimized.Mean_Absolute_Error];  % 1×num_images vector

% Plot
figure;
plot((echotimes(10:5:end)+108)/1e3, mae_values, '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Cutoff echo time [ms]');
ylabel('MAE [%]');
%ylim([2 5.5]);
set(gca, 'FontSize', 17);

exportgraphics(gcf, 'W:\radiologie\mrt-probanden\AG_Laun\Matthias Rohe\Daten\2025_07_13_NumberOfEchoes\2025_04_23_GTT\Bilder\R2_Results_Optimized.png', 'Resolution', 500);


%%
%%
% Extract MAE values from results struct
mae_values = [results.Mean_Absolute_Error];
mae_values_opt = [results_optimized.Mean_Absolute_Error];  % 1×num_images vector
%%
% Plot
figure;
hold on;
plot((echotimes(10:10:end)+50)/1e3, mae_values(1:end), '-o', 'LineWidth', 2, 'MarkerSize', 6);
plot((echotimes(10:10:end)+50)/1e3, mae_values_opt(1:end), '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Cutoff echo time [ms]');
ylabel('MAE [%]');
%ylim([2 5.5]);
legend('Non-optimized', 'Optimized');
set(gca, 'FontSize', 17);

exportgraphics(gcf, 'W:\radiologie\mrt-probanden\AG_Laun\Matthias Rohe\Daten\2025_07_13_NumberOfEchoes\2025_01_22_Phantom_PulseDuration\Bilder\Results_Both.png', 'Resolution', 500);

