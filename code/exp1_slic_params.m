% Experiment-1
% dataset: brainweb
% slic+lstm,superpixel value exp

%% dataset setting
brainwebFolder = 'C:\Users\shake\Desktop\MRBrainSeg\dataset\brainweb';
classes = [
    "BG"
    "CSF"
    "GM"
    "WM"
    ];
cmap = brainwebColorMap;
gtIDs = [1 2 3 4];

t1Dir = fullfile(brainwebFolder, 't1');
t2Dir = fullfile(brainwebFolder, 't2');
pdDir = fullfile(brainwebFolder, 'pd');
gtDir = fullfile(brainwebFolder, 'gt');

t1 = imageDatastore(t1Dir);
t2 = imageDatastore(t2Dir);
pd = imageDatastore(pdDir);
gt = pixelLabelDatastore(gtDir, classes, gtIDs);

[t1Train, t1Test, t2Train, t2Test, pdTrain, pdTest, gtTrain, gtTest] = partitionBrainWebData(t1, t2, pd, gt);
numTrain = numel(t1Train.Files);
numTest = numel(t1Test.Files);

% display sample images
% sampleIm = readimage(t1, 350);
% samplegt = readimage(gt, 350);
% sampleOverlay = labeloverlay(sampleIm, samplegt, 'ColorMap', cmap, 'Transparency',0.6);
% imshow(sampleOverlay);
% pixelLabelColorbar(cmap, classes);

% analyze dataset statistics
% analyzeGtData(gt, classes);
% plotClassHistogram(gtTrain, pdTrain);

%% method parameters setting
% superpixel parameters
% slicNum = [500:250:4000];
% compactness = [5 10 20];

% feature extraction mode
featureMode = 6;
numFeature = 3;

% network hyperparameters
numHiddenUnits = 40;
numClasses = 4;
maxEpochs = 10;
MiniBatchSize = 512;

% train and save models
% for compactness = 10
%     for slicNum = 2000
%         fprintf('training: compactness=%d, slicNum=%d\n', compactness, slicNum);
%         net = train(t1Train, t2Train, pdTrain, gtTrain, numTrain, slicNum, compactness, featureMode, numFeature, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%         net_name = ['model_slic_bilstm_' num2str(slicNum) '_' num2str(compactness) '_' num2str(featureMode) '_pn0.mat'];
%         save(net_name, 'net');
%     end
% end

% test on dataset and save results
for compactness = 10
    for slicNum = 2000
        fprintf('testing: compactness=%d, slicNum=%d\n', compactness, slicNum);
        net_name = ['model_slic_bilstm_' num2str(slicNum) '_' num2str(compactness) '_' num2str(featureMode) '_pn0.mat'];
        load('C:\Users\shake\Desktop\MRBrainSeg\trained model\model_slic_params_brainweb\model_slic_bilistm_2000_10_6.mat', 'net');
        [test_metric, superpixel_metric] = testAll(net, t1Test, t2Test, pdTest, gtTest, numTest, slicNum, compactness, featureMode, MiniBatchSize);
        test_metric_name = ['test_metric_slic_bilstm_' num2str(slicNum) '_' num2str(compactness) '_' num2str(featureMode) '_pn0.mat'];
        superpixel_metric_name = ['superpixel_metric_slic_bilstm_' num2str(slicNum) '_' num2str(compactness) '_' num2str(featureMode) '_pn0.mat'];
        save(test_metric_name, 'test_metric');
        save(superpixel_metric_name, 'superpixel_metric');
    end
end

%% One image test step1 
test1_Idx = 83;
[test1_t1, test1_info] = readimage(t1Test, test1_Idx);
test1_t2 = readimage(t2Test, test1_Idx);
test1_pd = readimage(pdTest, test1_Idx);

[test1_L, test1_N] = superpixels(test1_t1, 2000, 'Compactness', 10, 'NumIterations', 1);

% display superpixel oversegment images
dispSuperpixel(test1_t1, test1_L, test1_N, true, 4);

%%
[test1_L, test1_N] = removeBackground(test1_t1, test1_L, test1_N, 0.05);
test1_A = constructAMat(test1_L, test1_N);
test1_G = graph(test1_A);


% display superpixel graph
dispSuperpixelGraph(test1_t1, test1_L, test1_N, test1_G, 4);
% dispSubGraph(test1_t1, test1_L, test1_N, test1_G, 10, 20, 4);

%%
test1_X = createSeq(test1_t1, test1_t2, test1_pd, test1_G, test1_L, test1_N, featureMode);
test1_Y = createLabel(gtTest, test1_Idx, test1_L, test1_N);

test1_pred = classify(net, test1_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');

[test1_lbl, test1_lblImage] = creatPredImage(test1_Y, test1_L, test1_N);
test1_gt = readimage(gtTest, test1_Idx);
maxAcc = sum(sum(test1_lbl == double(test1_gt)))./numel(test1_lbl);
fprintf('superpixel accuracy: %.4f \n', maxAcc);

test1_metric = struct('classifyAcc',0,'Acc',0,'JS',{},'JS_avg',0,'DSC',{},'DSC_avg',0);
test1_metric(1).classifyAcc = sum(test1_pred == categorical(test1_Y))./numel(categorical(test1_Y));    % show classification accuracy
fprintf('classify accuracy: %.4f (%d/%d) \n', test1_metric(1).classifyAcc, sum(test1_pred == categorical(test1_Y)), numel(categorical(test1_Y)));

[test1_predlbl, test1_predImage] = creatPredImage(test1_pred, test1_L, test1_N);

%% One image test step2: display prediction image
% Display the result
test1_predImageOverlay = labeloverlay(test1_t1,test1_predlbl,'Colormap',cmap,'Transparency',0);
% figure;imshow(test1_predImageOverlay);
% pixelLabelColorbar(cmap, classes);
% imwrite(test1_predImage, 'C:\Users\shake\Desktop\专利附图\7-2.jpg');
% imwrite(label2rgb(double(test1_gt),cmap), 'C:\Users\shake\Desktop\专利附图\7-1.jpg');

% Compare with ground truth
% figure;imshowpair(uint8(test1_predlbl), uint8(test1_gt));
% imwrite(test1_predImageOverlay,'C:\Users\shake\Desktop\MRBrainSeg\figs\pred_c079_slic_bilstm_2000_10_f6.bmp');

% display with orginal image
test1_t1_ = imread('C:\Users\shake\Desktop\MRBrainSeg\dataset\brainweb\t1_\t076.bmp');
predImaget1 = createPredOrg(test1_predImage, test1_t1_, 0.6);
figure; imshow(predImaget1);
imwrite(predImaget1,'C:\Users\shake\Desktop\MRBrainSeg\figs\predt1_t076_slic_bilstm_2000_10_f6.bmp');

%% One image test step3: compute metrics

% Compute global Accuracy
test1_metric(1).Acc = sum(sum(test1_predlbl == double(test1_gt)))./numel(test1_predlbl);

% Compute iou(JS)
% jaccard(A,B) = | intersection(A,B) | / | union(A,B) |
test1_metric(1).JS = jaccard(test1_predlbl, double(test1_gt));
test1_metric(1).JS_avg = mean(test1_JS);

% Compute DSC
% dice(A,B) = 2 * | intersection(A,B) | / ( | A | + | B | )
test1_metric(1).DSC = dice(test1_predlbl, double(test1_gt));
test1_metric(1).DSC_avg = mean(test1_DSC)


%% 




