% Experiment
% dataset: brainweb
% slic+bilstm,

%% dataset setting
brainwebFolder = '/Users/shake/Desktop/MRBrainSeg/dataset/brainweb';
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
% superpixel parameters
slicNum = 2000;
compactness = 10;

% feature extraction mode
featureMode = 6;
numFeature = 3;

% network hyperparameters
% numHiddenUnits = [10:10:100];
numClasses = 4;
maxEpochs = 25;
MiniBatchSize = 512;

% train and save models
% for numHiddenUnits = 10:10:100
%     fprintf('training: numHiddenUnits=%d\n', numHiddenUnits);
%     net = train(t1Train, t2Train, pdTrain, gtTrain, numTrain, slicNum, compactness, featureMode, numFeature, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['model_slic_lstm_' num2str(numHiddenUnits) '.mat'];
%     save(net_name, 'net');
% end

% test on dataset and save results
for numHiddenUnits = 10:10:100
    fprintf('testing: numHiddenUnits=%d\n', numHiddenUnits);
    net_name = ['model_slic_lstm_' num2str(numHiddenUnits) '.mat'];
    load(net_name, 'net');
    [test_metric, superpixel_metric] = testAll(net, t1Test, t2Test, pdTest, gtTest, numTest, slicNum, compactness, featureMode, MiniBatchSize);
    test_metric_name = ['test_metric_slic_lstm_' num2str(numHiddenUnits) '.mat'];
    superpixel_metric_name = ['superpixel_metric_slic_lstm_' num2str(numHiddenUnits) '.mat'];
    save(test_metric_name, 'test_metric');
    save(superpixel_metric_name, 'superpixel_metric');
end