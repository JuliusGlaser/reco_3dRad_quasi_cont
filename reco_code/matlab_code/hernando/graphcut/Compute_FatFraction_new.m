function fatfraction = Compute_FatFraction_new(outParams)

fatfraction = zeros(size(outParams.species(1).amps));
fatfraction = zeros(fatfraction(1:3));

for slice = 1:size(outParams.species(1).amps, 3)
    water_image = outParams.species(1).amps(:,:,slice,1);
    fat_image = outParams.species(2).amps(:,:,slice,1);
    
    % Loop through each pixel
    for i = 1:size(water_image, 1)
        for j = 1:size(water_image, 2)
            if abs(fat_image(i, j)) >= abs(water_image(i,j))
                % Perform a calculation with image1's pixel
                fatfraction(i, j, slice) = abs(fat_image(i, j)) ./ (abs(water_image(i,j) + fat_image(i, j)));
            else
                % Perform a calculation with image2's pixel
                fatfraction(i, j, slice) = 1 - abs(water_image(i,j)) ./ (abs(water_image(i,j) + fat_image(i, j)));
            end
        end
    end
end