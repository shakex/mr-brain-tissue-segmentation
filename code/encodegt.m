gray = imageDatastore('E:\dataset\pets\bulldog_gt');
n = numel(gray.Files);
for i=1:n
    [img,info] = readimage(gray,i);
    [row, col] = size(img);
    name = info.Filename(28:end);
    encode = ones(row,col,'uint8');
    for m = 1:row
        for n = 1:col
            if img(m,n)==2
                encode(m,n)=0;
            elseif img(m,n)==1
                encode(m,n)=1;
            elseif img(m,n)==3
                encode(m,n)=2;
            end
        end
    end
    imwrite(encode, fullfile('E:\dataset\pets\bulldog_gt_', name));
end