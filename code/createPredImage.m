function [outlbl,outrgb] = createPredImage(pred, L, N)
% create prediction image with prediction cell

pixelIdxList = label2idx(L);
outlbl = ones(size(L,1),size(L,2));

for labelVal = 1:N
    outlbl(pixelIdxList{labelVal}) = pred(labelVal);
%     outlbl(pixelIdxList{labelVal}) = outlbl(pixelIdxList{labelVal}) - 1;
end

cmap = brainwebColorMap;
% cmap = petColorMap;
outrgb = label2rgb(outlbl, cmap);

end


