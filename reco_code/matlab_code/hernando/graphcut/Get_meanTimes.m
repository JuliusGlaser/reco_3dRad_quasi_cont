

total_pros = 20000;
slidingwindowpros = 10;
slidingwindowstep = 10;

echospacing = 1700;%params.acquisition.TEspacing; % [µs]
numCon = 6;%params.acquisition.echos;
pros = total_pros * numCon;%params.acquisition.projections * numCon; % [#]
steps = (pros - slidingwindowpros)/slidingwindowstep + 1;%(pros - params.slidingwindowpros)/params.slidingwindowstep + 1; 
echostep = echospacing / pros * numCon;  %echospacing / pros * numCon      echospacing / 65000

GRAD_RASTER_TIME = 10;        
T_raw = (0:pros-1) * echostep + 0.1*1e3;   
T_rounded = ceil(floor(T_raw) / GRAD_RASTER_TIME) * GRAD_RASTER_TIME; 
numBins = floor((length(T_rounded) - slidingwindowpros) / slidingwindowstep) + 1;  
meanTimes = zeros(1, numBins);
for i = 1:numBins
    startIdx = (i - 1) * slidingwindowstep + 1;
    endIdx = startIdx + slidingwindowpros - 1;
    meanTimes(i) = mean(T_rounded(startIdx:endIdx));
end