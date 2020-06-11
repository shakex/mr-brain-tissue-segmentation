function pxdsClass = createPixelLabel(pxds, idx)

labelImage = readimage(pxds,idx);
N = numel(labelImage);
[row, col] = size(labelImage);
pxdsClass = zeros(N,1);

for i = 1:row
    for j = 1:col
        pxdsClass((i-1)*col+j) = labelImage(i, j);
    end
end


end