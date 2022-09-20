function [AlignedThermal, AlignedMask] = Fusion(visible, thermal, tcp, vcp)
%FUSION Fusion of thermal and visible images
    TransformMatrix = fitgeotrans(tcp, vcp, 'projective');
    visibleRefObj = imref2d(size(visible));
    thermalRefObj = imref2d(size(thermal));
    mask = ones(size(thermal));
    
    AlignedThermal = imwarp(thermal, thermalRefObj, TransformMatrix, ...
        'OutputView', visibleRefObj, 'SmoothEdges', true);
    AlignedMask = imwarp(mask, thermalRefObj, TransformMatrix, ...
        'OutputView', visibleRefObj, 'SmoothEdges', true);
end

