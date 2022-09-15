
clc;
clear;
close all;
home;

kappa = 30;
amount = 0.5;
delta_t = 0.14;
edgeThreshold = 0.4;
viz_rate = 0.6;
th_rate = 0.3;

dataset_dir = "/home/phm/Datasets/Case Study  02_ Ulaval_Road (Handheld)-20220913T005457Z-001/";
dataset_viz = fullfile(dataset_dir, "visible");
dataset_ir = fullfile(dataset_dir, "thermal");

dataset_fused = fullfile(dataset_dir, "fused");
if not(isfolder(dataset_fused))
    mkdir(dataset_fused)
end

flist = dir(fullfile(dataset_viz,'*.jpg'));
num_data = length(flist);

for i = 0:num_data-1
    vfile = fullfile(dataset_viz, sprintf('%d_VIS.jpg',i));
    tfile = fullfile(dataset_ir, sprintf('%d_IR.jpg',i));
    ffile = fullfile(dataset_fused, sprintf('%d_Fused.jpg',i));

    if isfile(vfile) && isfile(tfile)
        fprintf("Processing image %d ...", i);
        visible = rgb2gray(imread(vfile));
        thermal = rgb2gray(imread(tfile));
        
        % Enhance the contrast around the edges
        thermal = localcontrast(thermal, edgeThreshold, amount);
        % Enhancing the image intensity
        thermal = imadjust(thermal);
        % Complement the thermal image
        thermal = imcomplement(thermal);

        fused = fusion_thviz(visible, thermal, edgeThreshold, amount, delta_t, kappa, viz_rate, th_rate);
        fused = fused ./ max(fused,[],'all');
        imwrite(fused, ffile);
        fprintf(" Done\n");
    else
        fprintf(" Failed\n")
    end
end