function pxds = resizeBrainwebPixelLabels(pxds, labelFolder)
% Resize pixel label data to [224 224].

classes = pxds.ClassNames;
labelIDs = 1:numel(classes);
% if ~exist(labelFolder,'dir')
%     mkdir(labelFolder)
% else
%     pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
%     return; % Skip if images already resized
% end

reset(pxds)
while hasdata(pxds)
    % Read the pixel data.
    [C,info] = read(pxds);
    
    % Convert from categorical to uint8.
    L = uint8(C);
    
    % Resize the data. Use 'nearest' interpolation to
    % preserve label IDs.
    [row, col] = size(L);
    new = ones(224,224,'uint8');
    for i = 1:row
        for j = 1:col
            new(i,j) = L(i,j);
        end
    end    
    
    % Write the data to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(new,[labelFolder filename ext])
end

labelIDs = 1:numel(classes);
pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
end