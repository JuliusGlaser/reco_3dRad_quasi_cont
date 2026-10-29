% Function: coilCombine
%
% Description: combine multi-coil image sequences
%
% Based on: Walsh DO, Gmitro AF, Marcellin MW. Adaptive reconstruction of
% phased array MR imagery. Magn Reson Med 2000;43:682-690
% 
% Parameters:
% im1: the multi-coil images (size [nx,ny,nz,ncoils,nTE])
%
% Returns: 
% im2: the coil-combined images (size [nx,ny,nz,1,nTE])
%
% Author: Diego Hernando
% Date created: August 13, 2011
% Date last modified: December 8, 2011

function im2 = coilCombine(im1)

% coilCombine - Adaptive multi-coil combination (Walsh 2000)
% Parallelized + memory-optimized version

% Permute to [x, y, TE, coil]
im1 = permute(im1, [1 2 5 4 3]);
[sx, sy, N, C] = size(im1);
filtsize = 7;

% --- Step 1: Compute local correlation matrices Rs
Rs = zeros(sx, sy, C, C, 'like', im1);

kernel = ones(filtsize, 'like', im1);
for kc1 = 1:C
    for kc2 = kc1:C  % symmetry: compute only half
        tmp = zeros(sx, sy, 'like', im1);
        for kn = 1:N
            tmp = tmp + im1(:,:,kn,kc1) .* conj(im1(:,:,kn,kc2));
        end
        tmp = filter2(kernel, tmp, 'same');
        Rs(:,:,kc1,kc2) = tmp;
        if kc1 ~= kc2
            Rs(:,:,kc2,kc1) = conj(tmp);
        end
    end
end

% --- Step 2: Compute and apply coil combination filter
im2 = zeros(sx, sy, 1, 1, N, 'like', im1);

for k = 1:(sx * sy)
    [kx, ky] = ind2sub([sx sy], k);
    Rlocal = squeeze(Rs(kx, ky, :, :));

    if any(isnan(Rlocal(:))) || all(Rlocal(:) == 0)
        continue;
    end

    % SVD for dominant coil sensitivity
    [U, ~] = svd(Rlocal, 'econ');
    myfilt = U(:,1);

    voxelData = squeeze(im1(kx, ky, :, :)).';  % [C x N]
    im2(kx, ky, 1, 1, :) = myfilt' * voxelData;
end

if isa(im1, 'single')
    im2 = single(im2);
end
end
