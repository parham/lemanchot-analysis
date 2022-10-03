function [regionsThermal, regionsMask, backgrndIndex, seg] = ThermalROIAnalysis(thermal, mask, level, regSize)
    %THERMALROIANALYSIS Summary of this function goes here
    %   Detailed explanation goes here

    %% Segmentation of image
    seg = imsegkmeans(single(thermal),level);

    %% Extract regions of K-mean
    regindexs = unique(seg);
    regindSize = length(regindexs);

    regionsThermal = [];
    regionsMask = [];
    regionsSize = [];
    for i = 1:regindSize
        rmask = mask;
        rmask(seg ~= i) = 0;
        % Extract independent components
        mks = bwconncomp(logical(rmask),8);
        for idx = 1:mks.NumObjects
            if length(mks.PixelIdxList{idx}) > regSize
                tmp = uint8(zeros(size(rmask)));
                tmp(mks.PixelIdxList{idx}) = 1;
                if sum(tmp,'all') > regSize
                    % Prepare and save the masks
                    regionsMask = cat(3,regionsMask,tmp);
                    rgsz = sum(tmp,'all');
                    regionsSize = cat(1,regionsSize,rgsz);
                    % Prepare and save the thermal regions
                    tr = thermal .* tmp;
                    regionsThermal = cat(3, regionsThermal, tr);
                end
            end
        end
    end

    [~, backgrndIndex] = max(regionsSize);
    specimenMask = regionsMask(:,:,backgrndIndex);
    figure('Name', 'Auto-Detection of the Inspection Surface'); 
    imshow(specimenMask)
end

