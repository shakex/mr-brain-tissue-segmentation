function [imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = fcmTest(brainwebFolder, testIdx, t1Test, gtTest, k, cmap)

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

[centers,U] = fcm(X, k);
[A, B] = max(U);
fcmIdx = B';
[centers,centerIdx] = sort(centers);
for i=1:size(fcmIdx,1)
    if fcmIdx(i) == centerIdx(1,1)
        fcmIdx(i) = 1;
    elseif fcmIdx(i) == centerIdx(2,1)
        fcmIdx(i) = 2;
    elseif fcmIdx(i) == centerIdx(3,1)
        fcmIdx(i) = 3;
    elseif fcmIdx(i) == centerIdx(4,1)
        fcmIdx(i) = 4;
    end
end

predlbl = reshape(fcmIdx,[row,col]);
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