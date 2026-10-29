function fatfraction = Compute_FatFraction(water_outParams, fat_outParams)

fatfraction = zeros(size(water_outParams.species(1).amps));

% Loop through each pixel
for i = 1:size(water_outParams.species(1).amps, 1)
    for j = 1:size(water_outParams.species(1).amps, 2)
        if abs(fat_outParams.species(2).amps(i, j)) >= abs(water_outParams.species(1).amps(i, j))
            % Perform a calculation with image1's pixel
            fatfraction(i, j) = abs(fat_outParams.species(2).amps(i, j)) ./ (abs(water_outParams.species(1).amps(i, j) + fat_outParams.species(2).amps(i, j)));
        else
            % Perform a calculation with image2's pixel
            fatfraction(i, j) = 1 - abs(water_outParams.species(1).amps(i, j)) ./ (abs(water_outParams.species(1).amps(i, j) + fat_outParams.species(2).amps(i, j)));
        end
    end
end