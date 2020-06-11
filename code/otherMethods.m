% Experiment
% dataset: brainweb/MRBrainS

%% dataset setting
brainwebFolder = 'C:/Users/shake/Desktop/MRBrainSeg/dataset/brainweb';
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
% sampleIm = readimage(t1, 2);
% samplegt = readimage(gt, 2);
% sampleOverlay = labeloverlay(sampleIm, samplegt, 'ColorMap', cmap, 'Transparency',0.6);
% imshow(sampleOverlay);
% pixelLabelColorbar(cmap, classes);

% analyze dataset statistics
% analyzeGtData(gt, classes);
% plotClassHistogram(gtTrain, pdTrain);

%% method parameters setting
% kmeans params
k = 4;
time = 0;
accAll = 0;
dscAll = 0;
% for testIdx = 1:10
for testIdx = 1:numTest
    startTime = clock;
    [I,info] = readimage(t1Test,testIdx);
    
    % add noise(Gaussian)
%     var = 0.01;
%     I = imnoise(I,'gaussian',0,var);
    
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
    predImage = label2rgb(predlbl, cmap);

    endTime = clock;
    time = time + etime(endTime,startTime);
    dscAll = dscAll + mean(dice(predlbl, double(lbl)));
    accAll = accAll + sum(sum(predlbl == double(lbl)))./numel(predlbl);

end


% end

timePerImage = time
accAll = accAll / numTest
dscAll = dscAll / numTest
% test on dataset and save results
% for numHiddenUnits = 10:10:100
%     fprintf('testing: numHiddenUnits=%d\n', numHiddenUnits);
%     net_name = ['model_slic_lstm_' num2str(numHiddenUnits) '.mat'];
%     load(net_name, 'net');
%     [test_metric, superpixel_metric] = testAll(net, t1Test, t2Test, pdTest, gtTest, numTest, slicNum, compactness, featureMode, MiniBatchSize);
%     test_metric_name = ['test_metric_slic_lstm_' num2str(numHiddenUnits) '.mat'];
%     superpixel_metric_name = ['superpixel_metric_slic_lstm_' num2str(numHiddenUnits) '.mat'];
%     save(test_metric_name, 'test_metric');
%     save(superpixel_metric_name, 'superpixel_metric');
% end

%% FCM
accAll = 0;
accAll = 0;
for testIdx = 1:numTest
    k = 4;
    fcmmodel = 'fcm';
    [test1_acc, test1_dsc, test1_predImage, predOverlayt1] = fcmTest(brainwebFolder, testIdx, t1Test, gtTest, k, cmap);
    
    accAll = accAll + test1_acc;
    dscAll = dscAll + test1_dsc;
end

accAll = accAll / numTest
dscAll = dscAll / numTest

%% gray2rgb
for i=1:numel(t1.Files)
    [img,info] = readimage(t1, i);
    rgb = cat(3, img, img, img);
    saveName = info.Filename(end-7:end);
    imwrite(rgb, fullfile('C:\Users\shake\Desktop\MRBrainSeg\dataset\brainweb\resize\t1rgb', saveName))
end


%% CNNs: segnet-d2d3/unet-d2d3/fcn8s
% resize imgs to 224x224
t1Dir_r = fullfile(brainwebFolder, 'resize','t1');
t1Dir_r_rgb = fullfile(brainwebFolder, 'resize','t1rgb');
t2Dir_r = fullfile(brainwebFolder, 'resize','t2');
pdDir_r = fullfile(brainwebFolder, 'resize','pd');
gtDir_r = fullfile(brainwebFolder, 'resize','gt');
t1_r = imageDatastore(t1Dir_r);
t1_r_rgb = imageDatastore(t1Dir_r_rgb);
t2_r = imageDatastore(t2Dir_r);
pd_r = imageDatastore(pdDir_r);
gt_r = pixelLabelDatastore(gtDir_r, classes, gtIDs);
[t1Train_r, t1Test_r, t1Train_r_rgb, t1Test_r_rgb, gtTrain_r, gtTest_r] = partitionBrainwebDataCNN(t1_r,t1_r_rgb,gt_r);

%% train 4 cnns
[fcn8s,unetd2,unetd3] = trainCNN(t1Train_r, t1Train_r_rgb, gtTrain_r);

% save 4 models
modelPath = 'C:/Users/shake/Desktop/MRBrainSeg/trained model/model_different_methods_brainweb';
metricPath = 'C:/Users/shake/Desktop/MRBrainSeg/test results/metric_different_methods_brainweb';

% save(fullfile(modelPath, 'model_segnetd2_150.mat'), 'segnetd2');
% save(fullfile(modelPath, 'model_segnetd3_150.mat'), 'segnetd3');
save(fullfile(modelPath, 'model_unetd2_150.mat'), 'unetd2');
save(fullfile(modelPath, 'model_unetd3_150.mat'), 'unetd3');
save(fullfile(modelPath, 'model_fcn8s_10.mat'), 'fcn8s');

%% testAll
% load(fullfile(modelPath, 'model_segnetd2_150.mat'), 'segnetd2');
% load(fullfile(modelPath, 'model_segnetd3_150.mat'), 'segnetd3');
% load(fullfile(modelPath, 'model_unetd2_150.mat'), 'unetd2');
% load(fullfile(modelPath, 'model_unetd3_150.mat'), 'unetd3');
load(fullfile(modelPath, 'model_fcn8s_10.mat'), 'fcn8s');

% test_metric_segnetd2 = testAllCNN(segnetd2, t1Test, t1Test_r, gtTest, numTest);
% test_metric_segnetd3 = testAllCNN(segnetd3, t1Test, t1Test_r, gtTest, numTest);
% test_metric_unetd2 = testAllCNN(unetd2, t1Test, t1Test_r, gtTest, numTest);
% test_metric_unetd3 = testAllCNN(unetd3, t1Test, t1Test_r, gtTest, numTest);
test_metric_fcn8s = testAllCNN(fcn8s, t1Test, t1Test_r_rgb, gtTest, numTest);

% save(fullfile(metricPath, 'test_metric_segnetd2_150.mat'), 'test_metric_segnetd2');
% save(fullfile(metricPath, 'test_metric_segnetd3_150.mat'), 'test_metric_segnetd3');
% save(fullfile(metricPath, 'test_metric_unetd2_150.mat'), 'test_metric_unetd2');
% save(fullfile(metricPath, 'test_metric_unetd3_150.mat'), 'test_metric_unetd3');
save(fullfile(metricPath, 'test_metric_fcn8s_10.mat'), 'test_metric_fcn8s');

%% LSTM/BiLSTM
% feature extraction mode
% featureMode = 6;

% network hyperparameters
% numHiddenUnits = 300;
numClasses = 4;
maxEpochs = 6;
MiniBatchSize = 512;

% set save path
modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\model_noise_bilstm_MRbrainS';
metricPath = 'C:\Users\shake\Desktop\MRBrainSeg\test results\metric_noise_bilstm_MRbrainS';

% train and save models
% for featureMode = 4
%     for numHiddenUnits = 40
%         fprintf('training: feature mode = %d, n=%d\n', featureMode, numHiddenUnits);
%         net = trainPixel(t1Train, t1irTrain, t2flairTrain, gtTrain, numTrain, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%         net_name = ['model_bilstm_f' num2str(featureMode) '_n' num2str(numHiddenUnits) '.mat'];
%         save(fullfile(modelPath, net_name), 'net');
%     end
% end

% test on dataset and save results
for featureMode = 4
    for numHiddenUnits = 40
        fprintf('testing: f=%d, n=%d, ', featureMode, numHiddenUnits);
%         net_name = ['model_bilstm_f' num2str(featureMode) '_n' num2str(numHiddenUnits) '.mat'];
%         load(fullfile(modelPath, net_name), 'net');
        test_metric = testAllPixel(net, t1Test, t1irTest, t2flairTest, gtTest, numTest, featureMode, MiniBatchSize);
        test_metric_name = ['test_metric_bilstm_f' num2str(featureMode) '_n' num2str(numHiddenUnits) '.mat'];
        save(fullfile(metricPath, test_metric_name), 'test_metric');
    end
end

%% SVM/KNN/decisionTree
featureMode = 2;
Array = [];

for trainIdx = 1:10
    fprintf('FeatureMode = %d: %d/%d\n', featureMode, trainIdx, numTrain);
    imt1 = readimage(t1Train, trainIdx);
    imt1ir = readimage(t1irTrain, trainIdx);
    imt2flair = readimage(t2flairTrain, trainIdx);
    labelImage = readimage(gtTrain, trainIdx);

    [row, col] = size(imt1);
    ins = zeros(numel(imt1), 4);
    for i=1:row
        for j=1:col
            ins((i - 1) * col + j, 1) = imt1(i, j); 
            ins((i - 1) * col + j, 2) = imt1ir(i, j); 
            ins((i - 1) * col + j, 3) = imt2flair(i, j);
            ins((i - 1) * col + j, 4) = labelImage(i, j);
        end
    end
    Array = [Array;ins];
end

pixelTrain = array2table(Array);

%% test All
model_name = {'knn', 'ld', 'svm', 'tree'};

% set save path
modelPath = '/Users/shake/Desktop/MRBrainSeg/trained model/model_different_methods_brainweb';
metricPath = '/Users/shake/Desktop/MRBrainSeg/test results/metric_different_methods_brainweb';

% test on dataset and save results
for featureMode = 1:2
    for modelIdx = 1:4
        fprintf('testing %s, feature mode = %d\n', model_name{modelIdx}, featureMode);
        net_name = ['model_' model_name{modelIdx} '_10_f' num2str(featureMode)];
%         load(fullfile(modelPath, [net_name '.mat']));
        net = get_variable_via_load(fullfile(modelPath, [net_name '.mat']));
        test_metric = testAllPixel2(net, t1Test, t2Test, pdTest, gtTest, numTest, featureMode);
        test_metric_name = ['test_metric_' model_name{modelIdx} '_10_f' num2str(featureMode) '.mat'];
        save(fullfile(metricPath, test_metric_name), 'test_metric');
    end
end

%% Neural networks
featureMode = 2;
Array = [];
target = [];

for trainIdx = 1:10
    fprintf('FeatureMode = %d: %d/%d\n', featureMode, trainIdx, numTrain);
    imt1 = readimage(t1Train, trainIdx);
    imt2 = readimage(t2Train, trainIdx);
    impd = readimage(pdTrain, trainIdx);
    labelImage = readimage(gtTrain,trainIdx);

    [row, col] = size(imt1);
    ins1 = zeros(3, numel(imt1));
    ins2 = zeros(4, numel(imt1));
    for i=1:row
        for j=1:col
            ins1(1, (i - 1) * col + j) = imt1(i, j); 
            ins1(2, (i - 1) * col + j) = imt2(i, j); 
            ins1(3, (i - 1) * col + j) = impd(i, j);
            ins2(labelImage(i, j), (i - 1) * col + j) = 1;
        end
    end
    Array = [Array ins1];
    target = [target ins2];
end


%% MRBRainS
mrbriansFolder = 'C:/Users/shake/Desktop/MRBrainSeg/dataset/MRBrainS';
classes = [
    "BG"
    "CSF"
    "GM"
    "WM"
    ];
cmap = brainwebColorMap;
gtIDs = [1 2 3 4];

t1Dir = fullfile(mrbriansFolder, 't1_');
t1irDir = fullfile(mrbriansFolder, 't1_ir_');
t2flairDir = fullfile(mrbriansFolder, 't2_flair_');
gtDir = fullfile(mrbriansFolder, 'gt');

t1 = imageDatastore(t1Dir);
t1ir = imageDatastore(t1irDir);
t2flair = imageDatastore(t2flairDir);
gt = pixelLabelDatastore(gtDir, classes, gtIDs);

[t1Train, t1Test, t1irTrain, t1irTest, t2flairTrain, t2flairTest, gtTrain, gtTest] = partitionBrainWebData(t1, t1ir, t2flair, gt);
numTrain = numel(t1Train.Files);
numTest = numel(t1Test.Files);

% display sample images
sampleIm = readimage(t1, 100);
samplegt = readimage(gt, 100);
sampleOverlay = labeloverlay(sampleIm, samplegt, 'ColorMap', cmap, 'Transparency',0.6);
imshow(sampleOverlay);
pixelLabelColorbar(cmap, classes);


%% one test
test1_Idx = 8;
[test1_t1, test1_info] = readimage(t1Test, test1_Idx);
test1_t2 = readimage(t2Test, test1_Idx);
test1_pd = readimage(pdTest, test1_Idx);

test1_X = createPixelSeq(test1_t1, test1_t2, test1_pd);
test1_Y = createPixelLabel(gtTest, test1_Idx);

test1_pred = classify(net, test1_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');

[test1_lbl, test1_lblImage] = creatPixelPredImage(test1_t1, test1_Y);

test1_gt = readimage(gtTest, test1_Idx);
maxAcc = sum(sum(test1_lbl == double(test1_gt)))./numel(test1_lbl);
fprintf('superpixel accuracy: %.4f \n', maxAcc);

test1_metric = struct('classifyAcc',0,'Acc',0,'JS',{},'JS_avg',0,'DSC',{},'DSC_avg',0);
test1_metric(1).classifyAcc = sum(test1_pred == categorical(test1_Y))./numel(categorical(test1_Y));    % show classification accuracy
fprintf('classify accuracy: %.4f (%d/%d) \n', test1_metric(1).classifyAcc, sum(test1_pred == categorical(test1_Y)), numel(categorical(test1_Y)));

[test1_predlbl, test1_predImage] = creatPixelPredImage(test1_t1, test1_pred);

%% One image test step2: display prediction image
% Display the result
test1_predImageOverlay = labeloverlay(test1_t1,test1_predlbl,'Colormap',cmap,'Transparency',0);
figure;imshow(test1_predImageOverlay);
pixelLabelColorbar(cmap, classes);

% Compare with ground truth 
figure;imshowpair(uint8(test1_predlbl), uint8(test1_gt));

%% One image test step3: compute metrics

% Compute global Accuracy
test1_metric(1).Acc = sum(sum(test1_predlbl == double(test1_gt)))./numel(test1_predlbl)

% % Compute iou(JS)
% % jaccard(A,B) = | intersection(A,B) | / | union(A,B) |
% test1_metric(1).JS = jaccard(test1_predlbl, double(test1_gt));
% test1_metric(1).JS_avg = mean(test1_JS);
%
% % Compute DSC
% % dice(A,B) = 2 * | intersection(A,B) | / ( | A | + | B | )
% test1_metric(1).DSC = dice(test1_predlbl, double(test1_gt));
% test1_metric(1).DSC_avg = mean(test1_DSC);
