function [outlbl,outrgb] = createPredImage(img, pred)

[row, col] = size(img);
outlbl = ones(row,col);

for i = 1:row
    for j = 1:col
        outlbl(i,j) = pred((i-1)*col+j);
    end
end

cmap = brainwebColorMap;
outrgb = label2rgb(outlbl, cmap);

end