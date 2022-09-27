
clc;
clear;
close all;
home;

%% Load Images
visible = rgb2gray(imread('/home/phm/Datasets/Case Study  02_ Ulaval_Road (Handheld)-20220913T005457Z-001/visible/155_VIS.jpg'));
thermal = rgb2gray(imread('/home/phm/Datasets/Case Study  02_ Ulaval_Road (Handheld)-20220913T005457Z-001/thermal/155_IR.jpg'));

%% Fusion of the two modalities
kappa = 30;
amount = 0.5;
delta_t = 0.14;
edgeThreshold = 0.4;
viz_rate = 0.6;
th_rate = 0.4;

% Enhance the contrast around the edges
thermal = localcontrast(thermal, edgeThreshold, amount);
% Enhancing the image intensity
thermal = imadjust(thermal);
% Complement the thermal image
thermal = imcomplement(thermal);

fused = fusion_thviz(visible, thermal, edgeThreshold, amount, delta_t, kappa, viz_rate, th_rate);
fused = fused ./ max(fused,[],'all');

%% Visualizing the fused image
fig = figure(); 
tlo = tiledlayout(fig, 2, 2);

ax = nexttile(tlo);
imshow(visible,'Parent',ax);
title(['', "Visible"]);

ax = nexttile(tlo);
imshow(thermal,'Parent',ax);
title(['', "Thermal"]);

ax = nexttile(tlo);
imshow(fused, [], 'Parent',ax)
title(['', "Fused"])

