gray = imageDatastore('C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR\gt_');
n = numel(gray.Files);
for i=1:n
    [img,info] = readimage(gray,i);
    [row, col] = size(img);
    name = info.Filename(end-9:end);
    encode = ones(row,col,'uint8');
    for m = 1:row
        for n = 1:col
            if img(m,n)==85
                encode(m,n)=2;
            elseif img(m,n)==170
                encode(m,n)=3;
            elseif img(m,n)==254
                encode(m,n)=4;
            end
        end
    end
    imwrite(encode, fullfile('C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR\gt',name));
end