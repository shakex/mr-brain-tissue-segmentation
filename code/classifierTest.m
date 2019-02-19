function [imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = classifierTest(brainwebFolder, model, test1_Idx, t1Test, t2Test, pdTest, gtTest, cmap)
Array = [];
[test1_t1, test1_Info] = readimage(t1Test, test1_Idx);
test1_t2 = readimage(t2Test, test1_Idx);
test1_pd = readimage(pdTest, test1_Idx);
imagename = test1_Info.Filename(end-7:end-4);

% add gaussian noise
% test1_t1 = imnoise(test1_t1,'gaussian',0,var);
% test1_t2 = imnoise(test1_t2,'gaussian',0,var);
% test1_pd = imnoise(test1_pd,'gaussian',0,var);

[row, col] = size(test1_t1);
ins = zeros(numel(test1_t1), 3);
for i=1:row
    for j=1:col
        ins((i - 1) * col + j, 1) = test1_t1(i, j); 
        ins((i - 1) * col + j, 2) = test1_t2(i, j); 
        ins((i - 1) * col + j, 3) = test1_pd(i, j);
    end
end
Array = [Array;ins];
pixelTest = array2table(Array);

pred = model.predictFcn(pixelTest);
predlbl = reshape(pred,[col,row]);
predlbl = predlbl';

lbl = readimage(gtTest,test1_Idx);
% test1_predImage
test1_predImage = label2rgb(predlbl, cmap);
% predOverlay
test1_t1_ = imread(fullfile(brainwebFolder, 't1_', test1_Info.Filename(end-7:end)));
predOverlayt1 = createPredOrg(test1_predImage, test1_t1_, 0.6);
% test1_acc
test1_acc = sum(sum(predlbl == double(lbl)))./numel(predlbl);
test1_dsc = mean(dice(predlbl, double(lbl)));

end