
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

dataset_dir = "/home/phm/SSH_Drive/lemanchot-analysis/UL-Road-HandHeld/worst_case";
dataset_viz = fullfile(dataset_dir, "visible");
dataset_ir = fullfile(dataset_dir, "thermal");

dataset_fused = fullfile(dataset_dir, "fused_classics");
if not(isfolder(dataset_fused))
    mkdir(dataset_fused)
end

flist = dir(fullfile(dataset_viz,'*.jpg'));
num_data = length(flist);

for i = 1:num_data
    vfile = fullfile(dataset_viz, flist(i).name);
    tfile = fullfile(dataset_ir, strrep(flist(i).name, 'VIS', 'IR'));
    ffile = fullfile(dataset_fused, strrep(flist(i).name, 'VIS', 'Fused'));

    if isfile(vfile) && isfile(tfile)
        fprintf("Processing image %d ...", i);
        visible = rgb2gray(imread(vfile));
        thermal = rgb2gray(imread(tfile));
        
        % Enhance the contrast around the edges
        %thermal = localcontrast(thermal, edgeThreshold, amount);
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