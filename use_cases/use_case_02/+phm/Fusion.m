function [AlignedThermal, AlignedMask, TransformMatrix] = Fusion(visible, thermal, mask, tcp, vcp)
%FUSION Fusion of thermal and visible images
    TransformMatrix = fitgeotrans(tcp, vcp, 'projective');
    visibleRefObj = imref2d(size(visible));
    thermalRefObj = imref2d(size(thermal));
    maskRefObj = imref2d(size(mask));
    
    AlignedThermal = imwarp(thermal, thermalRefObj, TransformMatrix, ...
        'OutputView', visibleRefObj, 'SmoothEdges', true);
%     AlignedMask = imwarp(mask, thermalRefObj, TransformMatrix, ...
%         'OutputView', visibleRefObj, 'SmoothEdges', true);
    AlignedMask = imwarp(mask, maskRefObj, TransformMatrix, ...
        'OutputView', visibleRefObj, 'SmoothEdges', true);
    
end

