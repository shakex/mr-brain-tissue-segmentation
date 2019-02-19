% ablation study
% bilstm(f1,f2,f3,f4)
% slic+bilstm(f1,f2,f3,f4)

%% BrainWeb
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

% superpixel parameters
slicNum = 2000;
compactness = 10;

% feature extraction mode
% featureMode = 6;
numFeature = 3;

% network hyperparameters
numHiddenUnits = 40;
numClasses = 4;
maxEpochs = 20;
MiniBatchSize = 512;

% set path
modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\ablation study_brainweb';

% training: slic+bilstm
% for featureMode = 1:4
%     fprintf('training slic+bilstm: f=%d\n', featureMode);
%     net = train(t1Train, t2Train, pdTrain, gtTrain, numTrain, slicNum, compactness, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['slic_bilstm_f' num2str(featureMode) '.mat'];
%     save(fullfile(modelPath,net_name), 'net');
% end

% training: bilstm
% maxEpochs = 1;
% for featureMode = 1:4
%     fprintf('training bilstm: f=%d\n', featureMode);
%     net = trainPixel(t1Train, t2Train, pdTrain, gtTrain, numTrain, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['bilstm_f' num2str(featureMode) '.mat'];
%     save(fullfile(modelPath,net_name), 'net');
% end

% testing: slic+bilstm
% for featureMode = 1:4
%     fprintf('testing slic+bilstm: f=%d\n', featureMode);
%     net_name = ['slic_bilstm_f' num2str(featureMode) '.mat'];
%     load(fullfile(modelPath, net_name), 'net');
%     [test_metric, superpixel_metric] = testAll(net, t1Test, t2Test, pdTest, gtTest, numTest, slicNum, compactness, featureMode, MiniBatchSize);
%     test_metric_name = ['brainweb_metric_slic_bilstm_f' num2str(featureMode) '.mat'];
%     superpixel_metric_name = ['brainweb_superpixel_slic_bilstm_f' num2str(featureMode) '.mat'];
%     save(fullfile(modelPath, test_metric_name), 'test_metric');
%     save(fullfile(modelPath, superpixel_metric_name), 'superpixel_metric');
% end
% 
% % testing: bilstm
for featureMode = 1:4
    fprintf('testing bilstm: f=%d\n', featureMode);
    net_name = ['bilstm_f' num2str(featureMode) '.mat'];
    load(fullfile(modelPath, net_name), 'net');
    test_metric = testAllPixel(net, t1Test, t2Test, pdTest, gtTest, numTest, featureMode, MiniBatchSize);
    test_metric_name = ['brainweb_metric_bilstm_f' num2str(featureMode) '.mat'];
    save(fullfile(modelPath, test_metric_name), 'test_metric');
end


%% MRBrainS
mrbrainsFolder = 'C:\Users\shake\Desktop\MRBrainSeg\dataset\MRBrainS';
classes = [
    "BG"
    "CSF"
    "GM"
    "WM"
    ];
cmap = brainwebColorMap;
gtIDs = [1 2 3 4];

t1Dir = fullfile(mrbrainsFolder, 't1');
t1irDir = fullfile(mrbrainsFolder, 't1ir');
t2flairDir = fullfile(mrbrainsFolder, 't2flair');
gtDir = fullfile(mrbrainsFolder, 'gt');

t1 = imageDatastore(t1Dir);
t1ir = imageDatastore(t1irDir);
t2flair = imageDatastore(t2flairDir);
gt = pixelLabelDatastore(gtDir, classes, gtIDs);

[t1Train, t1Test, t1irTrain, t1irTest, t2flairTrain, t2flairTest, gtTrain, gtTest] = partitionBrainWebData(t1, t1ir, t2flair, gt);
numTrain = numel(t1Train.Files);
numTest = numel(t1Test.Files);

% superpixel parameters
slicNum = 2000;
compactness = 10;

% network hyperparameters
numHiddenUnits = 40;
numClasses = 4;
maxEpochs = 40;
MiniBatchSize = 512;

% set path
modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\ablation study_MRbrainS';

% training: slic+bilstm
% for featureMode = 1:4
%     fprintf('training slic+bilstm: f=%d\n', featureMode);
%     net = train(t1Train, t1irTrain, t2flairTrain, gtTrain, numTrain, slicNum, compactness, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['slic_bilstm_f' num2str(featureMode) '.mat'];
%     save(fullfile(modelPath, net_name), 'net');
% end

% training: bilstm
% maxEpochs = 1;
% for featureMode = 1:4
%     fprintf('training bilstm: f=%d\n', featureMode);
%     net = trainPixel(t1Train, t1irTrain, t2flairTrain, gtTrain, numTrain, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['bilstm_f' num2str(featureMode) '.mat'];
%     save(fullfile(modelPath, net_name), 'net');
% end

% testing: slic+bilstm
% for featureMode = 1:4
%     fprintf('testing slic+bilstm: f=%d\n', featureMode);
%     net_name = ['slic_bilstm_f' num2str(featureMode) '.mat'];
%     load(fullfile(modelPath, net_name), 'net');
%     [test_metric, superpixel_metric] = testAll(net, t1Test, t1irTest, t2flairTest, gtTest, numTest, slicNum, compactness, featureMode, MiniBatchSize);
%     test_metric_name = ['mrbrains_metric_slic_bilstm_f' num2str(featureMode) '.mat'];
%     superpixel_metric_name = ['mrbrains_superpixel_slic_bilstm_f' num2str(featureMode) '.mat'];
%     save(fullfile(modelPath, test_metric_name), 'test_metric');
%     save(fullfile(modelPath, superpixel_metric_name), 'superpixel_metric');
% end

% testing: bilstm
for featureMode = 1:4
    fprintf('testing bilstm: f=%d\n', featureMode);
    net_name = ['bilstm_f' num2str(featureMode) '.mat'];
    load(fullfile(modelPath, net_name), 'net');
    test_metric = testAllPixel(net, t1Test, t1irTest, t2flairTest, gtTest, numTest, featureMode, MiniBatchSize);
    test_metric_name = ['mrbrains_metric_bilstm_f' num2str(featureMode) '.mat'];
    save(fullfile(modelPath, test_metric_name), 'test_metric');
end
