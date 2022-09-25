
clear;
clc;
close all;

%% Parameters initialization
fontSize = 16;
root_dir = '/home/phm/Dataset/texture-analysis/piping_inspection';
vis_files = dir(fullfile(root_dir, 'visible', '*.jpg'));
% Create the fusion folder
fused_fpath = fullfile(root_dir, "fused");
if ~exist(fused_fpath, 'dir')
    fused_dir = mkdir(fused_fpath);
end

for findex = 1:length(vis_files)
    fprintf('Loading %s ... ',vis_files(findex).name);
    % Set the folder names
    vfname = vis_files(findex).name;

    fused_savefile = fullfile(fused_fpath, strrep(strrep(vfname, '.jpg', '.mat'), '_VIS', '_FUSED'));
    if isfile(fused_savefile)
        fprintf('Exist!\n')
        continue;
    end

    tfname = strrep(vfname, '_VIS', '_IR');
    vfile = fullfile(root_dir, 'visible', vfname);
    tfile = fullfile(root_dir, 'thermal', tfname);
    % Check if all modalities exist
    if ~isfile(vfile) || ~isfile(tfile)
        fprintf('Skipped!\n')
        continue;
    else
        fprintf('Done!\n')
    end
    
    % Read modalities
    viz = imread(vfile);
    ir = imread(tfile);
    % Manual multi-modal registration
    [tcp, vcp] = cpselect(ir, viz, 'Wait', true);
    % Fuse the aligned modalities
    if isempty(tcp) || isempty(vcp)
        fprintf('Registration is skipped ... %s\n', vfname);
        continue;
    else
        [aligned_ir, aligned_mask] = phm.Fusion(viz, ir, tcp, vcp);
    end

    imshow(imfuse(viz, aligned_ir, 'blend'), [])
    axis on;
    title('Original Grayscale Image', 'FontSize', fontSize);
    set(gcf, 'Position', get(0,'Screensize'));
    hFH = drawfreehand();

    status = false;
    if (~isempty(hFH) && isprop(hFH, 'Position'))
        if (size(hFH.Position,1) > 3)
            % Get the xy coordinates of where they drew.
            xy = hFH.Position;
            % get rid of imfreehand remnant.
            delete(hFH);
            % Overlay what they drew onto the image.
            hold on; % Keep image, and direction of y axis.
            % Extract Coordinates
            xCoordinates = xy(:, 1);
            yCoordinates = xy(:, 2);
            plot(xCoordinates, yCoordinates, 'ro', 'LineWidth', 2, 'MarkerSize', 10);
            caption = sprintf('Select the region');
            title(caption, 'FontSize', fontSize);
            hold off;
            % Extract the selected polygonal region of interest
            ROIMask = roipoly(viz, xCoordinates, yCoordinates);
            % Extract thermal ROI
            tmask = aligned_ir .* uint8(ROIMask);
            % Extract mask ROI
            mask = aligned_mask .* ROIMask;
            % Extract visible ROI
            vmask = viz .* uint8(ROIMask);
            
            okind = find(mask > 0);
            [ii,jj] = ind2sub(size(mask), okind);
            ymin = min(ii); ymax = max(ii); xmin = min(jj); xmax = max(jj);
            % Crop the thermal image to bound the selected ROI
            ir_roi = imcrop(tmask, [xmin,ymin,xmax-xmin+1,ymax-ymin+1]);
            % Crop the visible image to bound the selected ROI
            viz_roi = imcrop(vmask, [xmin,ymin,xmax-xmin+1,ymax-ymin+1]);
            % Crop the mask image to bound the selected ROI
            mask = imcrop(mask, [xmin,ymin,xmax-xmin+1,ymax-ymin+1]);
            
            fprintf('Saving the fused file ... %s\n',fused_savefile);
            save(fused_savefile, 'aligned_ir', 'viz', 'ir_roi', 'viz_roi');
        end
    else
        fprintf('ROI Selection is skipped ... %s\n', vfname);
    end
end
