function [imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = kmeansTest(brainwebFolder, testIdx, t1Test, gtTest, k, cmap)

[I, test1_Info] = readimage(t1Test,testIdx);
imagename = test1_Info.Filename(end-7:end-4);
% add gaussian noise
% I = imnoise(I,'gaussian',0,var);

[row, col] = size(I);
X = zeros(numel(I), 1);
for j = 1:col
    for i = 1:row
       X((j - 1) * row + i,1) = I(i,j);
    end
end

[kmeansIdx,centroid] = kmeans(X, k,'Distance','sqeuclidean');

[centroid,centroidIdx] = sort(centroid);
for i=1:size(kmeansIdx,1)
    if kmeansIdx(i) == centroidIdx(1,1)
        kmeansIdx(i) = 1;
    elseif kmeansIdx(i) == centroidIdx(2,1)
        kmeansIdx(i) = 2;
    elseif kmeansIdx(i) == centroidIdx(3,1)
        kmeansIdx(i) = 3;
    elseif kmeansIdx(i) == centroidIdx(4,1)
        kmeansIdx(i) = 4;
    end
end

predlbl = reshape(kmeansIdx,[row,col]);
lbl = readimage(gtTest,testIdx);
% test1_predImage
test1_predImage = label2rgb(predlbl, cmap);
% predOverlay
test1_t1_ = imread(fullfile(brainwebFolder, 't1_', test1_Info.Filename(end-7:end)));
predOverlayt1 = createPredOrg(test1_predImage, test1_t1_, 0.6);
% test1_acc
test1_acc = sum(sum(predlbl == double(lbl)))./numel(predlbl);
test1_dsc = mean(dice(predlbl, double(lbl)));

end