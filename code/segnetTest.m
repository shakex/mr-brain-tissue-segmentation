function [imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = segnetTest(brainwebFolder, net, test1_Idx, t1Test, gtTest, cmap)

[test1_t1_org, test1_Info] = readimage(t1Test, test1_Idx);
imagename = test1_Info.Filename(end-7:end-4);
imagename2 = test1_Info.Filename(end-7:end);


test1_t1 = imread(fullfile('/Users/shake/Documents/master/project/MRBrainSeg/dataset/brainweb/resize/t1',imagename2));
predlbl = semanticseg(test1_t1, net);
predlbl = resize2orginal(predlbl, test1_t1_org);

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