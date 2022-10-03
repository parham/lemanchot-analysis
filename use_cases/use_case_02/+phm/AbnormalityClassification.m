function [problematicIndexs] = AbnormalityClassification(img, regionsMask, maskTexture, backgrndIndex, subRatio)
%ABNORMALITYCLASSIFICATION Abnormality classification
%   Detailed explanation goes here
%% Analyze the defects with extracted texture
% subRatio = 0.5;
numRegions = size(regionsMask,3);
problematicIndexs = zeros(1,numRegions);
for i = 1:numRegions
    if i == backgrndIndex
        continue;
    end
    rmsk = logical(regionsMask(:,:,i));
    tmp = and(maskTexture,rmsk);
    ratio = sum(tmp,'all') / sum(rmsk,'all');
    if ratio >= subRatio
        problematicIndexs(i) = 1;
    end
end

figure('Name', 'Abnormality Analysis Result');
imshow(img);
hold on
for i = 1:numRegions
    color = 'b';
    if problematicIndexs(i) == 1
        color = 'r';
    end
    if i ~= backgrndIndex
        bb = regionprops(regionsMask(:,:,i), 'BoundingBox');
        pos = [bb.BoundingBox(1),bb.BoundingBox(2),bb.BoundingBox(3),bb.BoundingBox(4)];
        rectangle('Position', pos, ...
            'EdgeColor', color, 'LineWidth', 3);
        text(pos(1),pos(2), strcat('OBJ #',num2str(i)), 'HorizontalAlignment', 'left', 'Color', 'y');
    end
end
hold off
end

