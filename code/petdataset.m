petFolder = 'C:/Users/shake/Desktop/MRBrainSeg/dataset/pets';
classes = [
    "Foreground"
    "Background"
    "Notclassified"
    ];
cmap = petColorMap;
gtIDs = [1 2 3];

imgDir = fullfile(petFolder, 'bulldog');
gtDir = fullfile(petFolder, 'bulldog_gt');

img = imageDatastore(imgDir);
gt = pixelLabelDatastore(gtDir, classes, gtIDs);

% display sample images
% sampleIdx = 5;
% sampleIm = readimage(img, sampleIdx);
% samplegt = readimage(gt, sampleIdx);
% sampleOverlay = labeloverlay(sampleIm, samplegt, 'ColorMap', cmap, 'Transparency',0.4);
% imshow(sampleOverlay);
% pixelLabelColorbar(cmap, classes);

[imgTrain, imgTest, gtTrain, gtTest] = partitionPetData(img, gt);
numTrain = numel(imgTrain.Files);
numTest = numel(imgTest.Files);

%%
cmap = petColorMap;
% superpixel parameters
slicNum = 1000;
compactness = 10;

% feature extraction mode
numFeature = 7;

% network hyperparameters
numHiddenUnits = 40;
numClasses = 3;
maxEpochs = 40;
MiniBatchSize = 512;

% set save path
modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\model_noise_bilstm_MRbrainS\new';
metricPath = 'C:\Users\shake\Desktop\MRBrainSeg\test results\metric_noise_bilstm_MRbrainS\new';

XTrain = {};
YTrain = {};

for trainIdx = 1:numTrain
    fprintf('%d/%d \t', trainIdx, numTrain);
    im = readimage(imgTrain, trainIdx);
    [L, N] = superpixels(im, slicNum, 'Compactness', compactness);    % slic
    
    % display superpixel oversegment images
%     dispSuperpixel(im, L, N, true, 1);
    
    A = constructAMat(L, N);
    G = graph(A);
    
    % display superpixel graph
%     dispSuperpixelGraph(im, L, N, G, 1);
    
    % feature
    lab = rgb2lab(im);
    hsv = rgb2hsv(im);
    [VG,A,PPG] = colorgrad(im);
%     figure;imshow(PPG);
    
    tic
    trainSeq = createSeqRGB(im,lab,hsv,PPG,G,L,N);
%     trainSeq = createSeq(imt1, imt2, impd, G, L, N, featureMode);   % extract features
    toc
    
    gtTrainClass = createLabel(gtTrain, trainIdx, L, N);    % prepare gt
%     [predlbl, predImage] = creatPredImage(gtTrainClass, L, N);
%     groundtruth = readimage(gtTrain, trainIdx);
%     figure;
%     subplot(1,2,1);imshow(predImage);title('slic gt');
%     subplot(1,2,2);imshow(label2rgb(double(groundtruth),cmap));title('ground truth');
    
    XTrain = [XTrain;trainSeq];
    YTrain = [YTrain;categorical(gtTrainClass)];
    
end

%% training configuration
layers = [ ...
    sequenceInputLayer(numFeature)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options2 = trainingOptions('adam',...
    'GradientThreshold',1, ...
    'GradientDecayFactor',0.95,...
    'SquaredGradientDecayFactor',0.999,...
    'InitialLearnRate',0.005,...
    'LearnRateSchedule','piecewise',... 
    'LearnRateDropFactor',0.8,... 
    'LearnRateDropPeriod',10,... 
    'MaxEpochs',maxEpochs,... 
    'MiniBatchSize',MiniBatchSize,...
    'Verbose',false,...
    'Plots','training-progress');

%% start training
net = trainNetwork(XTrain,YTrain,layers,options2);

%% test one
test1_Idx = 4;
test1_im = readimage(imgTest, test1_Idx);
tic
[test1_L, test1_N] = superpixels(test1_im, slicNum, 'Compactness', compactness);

test1_A = constructAMat(test1_L, test1_N);
test1_G = graph(test1_A);

% feature
test1_lab = rgb2lab(test1_im);
test1_hsv = rgb2hsv(test1_im);
[test1_VG,test1_AA,test1_PPG] = colorgrad(im);

test1_X = createSeqRGB(test1_im,test1_lab,test1_hsv,test1_PPG,test1_G,test1_L,test1_N);

test1_Y = createLabel(gtTest, test1_Idx, test1_L, test1_N);

test1_pred = classify(net, test1_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');
[test1_predlbl, test1_predImage] = creatPredImage(test1_pred, test1_L, test1_N);
toc

[test1_lbl, test1_lblImage] = creatPredImage(test1_Y, test1_L, test1_N);
test1_gt = readimage(gtTest, test1_Idx);
maxAcc = sum(sum(test1_lbl == double(test1_gt)))./numel(test1_lbl);
fprintf('superpixel accuracy: %.4f \n', maxAcc);

test1_metric = struct('classifyAcc',0,'Acc',0,'JS',{},'JS_avg',0,'DSC',{},'DSC_avg',0);
test1_metric(1).classifyAcc = sum(test1_pred == categorical(test1_Y))./numel(categorical(test1_Y));    % show classification accuracy
fprintf('classify accuracy: %.4f (%d/%d) \n', test1_metric(1).classifyAcc, sum(test1_pred == categorical(test1_Y)), numel(categorical(test1_Y)));

figure;
subplot(1,3,1);imshow(test1_predImage);title('prediction');
subplot(1,3,2);imshow(test1_lblImage);title('slic gt');
subplot(1,3,3);imshow(label2rgb(double(test1_gt),cmap));title('ground truth');

