
function [textureTop, textureBottom, maskTexture] = AbnormalityAnalysis(img, binThresh, maxNumPixs)
    %% Create Texture Image
    E = entropyfilt(img);
    S = stdfilt(img,ones(9));
    R = rangefilt(img,ones(9));
    Eim = rescale(E);
    Sim = rescale(S);
    figure; montage({Eim,Sim,R},'Size',[1 3],'BackgroundColor','w',"BorderSize",20)
    title('Texture Images Showing Local Entropy, Local Standard Deviation, and Local Range')
    % Create Mask for Bottom Texture
    BW1 = imbinarize(Eim, binThresh);
    % Create Mask for Bottom Texture
    BWao = bwareaopen(BW1, maxNumPixs);
    % Create Closed Texture Image
    nhood = ones(9);
    closeBWao = imclose(BWao,nhood);
    % Create Mask of Bottom Texture
    maskTexture = imfill(closeBWao,'holes');
    
    figure('Name','Texture Images');
    subplot(2,2,1);
    imshow(BW1);
    title('STEP 1 : Thresholded Texture Image');
    subplot(2,2,2);
    imshow(BWao);
    title('STEP 2 : Area-Opened Texture Image')
    subplot(2,2,3);
    imshow(closeBWao);
    title('STEP 3 : Closed Texture Image');
    subplot(2,2,4);
    imshow(maskTexture);
    title('STEP 4 : Mask of Bottom Texture')
   
    %% Use Mask to Segment Textures
    textureTop = img;
    textureTop(maskTexture) = 0;
    textureBottom = img;
    textureBottom(~maskTexture) = 0;
    figure; montage({textureTop,textureBottom},'Size',[1 2],'BackgroundColor','w',"BorderSize",20)
    title('Segmented Top Texture (Left) and Segmented Bottom Texture (Right)')
    %% Display Segmentation Results
    L = maskTexture+1;
    figure; imshow(labeloverlay(img,L))
    title('Labeled Segmentation Regions')
    
end
