function [fused] = fusion_thviz(visible, thermal, edgeThreshold, amount, delta_t, kappa, viz_rate, th_rate)
%FUSION_THVIZ The fusion of thermal and visible images to enhance the
%visiblity of defects.
% Adopted from https://www.mathworks.com/matlabcentral/fileexchange/63591-fusion-of-infrared-and-visible-sensor-images-based-on-anisotropic-diffusion-and-kl-transform
%
% Default values:
%   edgeThreshold = 0.4
%   amount = 0.5
%   delta_t = 0.12
%   kappa = 30

%% Preprocessing modalities

if size(visible,3) == 3
    viz = rgb2gray(visible);
else
    viz = visible;
end

if size(thermal,3) == 3 
    ir = rgb2gray(thermal);
else
    ir = thermal;
end

%% Calculating Anisotropic Diffusion
num_iter = 10;
option = 1;

tic

Aviz = anisodiff2D(viz,num_iter,delta_t,kappa,option);
Air = anisodiff2D(ir,num_iter,delta_t,kappa,option);

Dviz = double(viz) - Aviz;
Dir = double(ir) - Air;

Cfused = cov([Dviz(:) Dir(:)]);

[V11, D11] = eig(Cfused);
if D11(1,1) >= D11(2,2)
    pca1 = V11(:,1) ./ sum(V11(:,1));
else
    pca1 = V11(:,2) ./ sum(V11(:,2));
end

im_part1 = pca1(1) * Dviz + pca1(2) * Dir;
im_part2 = (viz_rate * Aviz + th_rate * Air);

fused = (double(im_part1) + double(im_part2));

end

