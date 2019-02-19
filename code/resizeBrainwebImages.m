function imds = resizeBrainwebImages(imds, imageFolder)
% Resize images to [224 224].
% 
% if ~exist(imageFolder,'dir') 
%     mkdir(imageFolder)
% else
%     imds = imageDatastore(imageFolder);
%     return; % Skip if images already resized
% end

reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);     
    
    % Resize image.
    [row, col] = size(I);
    new = zeros(224,224,'uint8');
    for i = 1:row
        for j = 1:col
            new(i,j) = I(i,j);
        end
    end    
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(new,[imageFolder filename ext])
end

imds = imageDatastore(imageFolder);
end

