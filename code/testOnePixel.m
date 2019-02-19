function [imagename, test1_acc, test1_dsc, test1_predImage, gtimage, predOverlayt1, gtOverlayt1] = testOnePixel(cmap, brainwebFolder, net, test1_Idx, t1Test, t2Test, pdTest, gtTest, featureMode, MiniBatchSize)
[test1_t1, test1_Info] = readimage(t1Test, test1_Idx);
test1_t2 = readimage(t2Test, test1_Idx);
test1_pd = readimage(pdTest, test1_Idx);
imagename = test1_Info.Filename(end-7:end-4);

% add nosie (gaussian)
% test1_t1 = imnoise(test1_t1,'gaussian',0, var);
% test1_t2 = imnoise(test1_t2,'gaussian',0, var);
% test1_pd = imnoise(test1_pd,'gaussian',0, var);

test1_X = createPixelSeq(test1_t1, test1_t2, test1_pd, featureMode);
% test1_Y = createPixelLabel(gtTest, test1_Idx);

test1_pred = classify(net, test1_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');

test1_gt = readimage(gtTest, test1_Idx);
[test1_predlbl, test1_predImage] = createPixelPredImage(test1_t1, test1_pred);

test1_t1_ = imread(fullfile(brainwebFolder, 't1_', test1_Info.Filename(end-7:end)));
predOverlayt1 = createPredOrg(test1_predImage, test1_t1_, 0.7);
gtOverlayt1 = createPredOrg(label2rgb(double(test1_gt),cmap), test1_t1_, 0.7);
gtimage = label2rgb(double(test1_gt),cmap);

test1_acc = sum(sum(test1_predlbl == double(test1_gt)))./numel(test1_predlbl);
test1_dsc = mean(dice(test1_predlbl, double(test1_gt)));
