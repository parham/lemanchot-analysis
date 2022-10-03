
clear;
clc;

seg_file = "/home/phm/Dataset/texture-analysis/piping_inspection/segs/23_FUSED.png";
batch_file = "/home/phm/Dataset/texture-analysis/piping_inspection/fused/23_FUSED.mat";

seg = imread(seg_file);
load(batch_file);
ir_roi = rgb2gray(ir_roi);
viz_roi = rgb2gray(viz_roi);
seg = imresize(seg, size(ir_roi));

[textureTop, textureBottom, maskTexture] = phm.AbnormalityAnalysis(viz_roi, 0.7, 250);
[regionsThermal, regionsMask, backgrndIndex, seg] = phm.ThermalROIAnalysis(ir_roi, seg, 5, 200);

newRegionMask = [];
nummask = size(regionsMask, 3);
for i = 1:nummask
    mm = regionsMask(:,:,i);
    sumregion = sum(mm,"all");
    if sumregion > 1000
        newRegionMask = cat(3,newRegionMask,mm);
    end
end

[problematicIndexs] = phm.AbnormalityClassification(ir_roi, newRegionMask, maskTexture, backgrndIndex, 0.5);