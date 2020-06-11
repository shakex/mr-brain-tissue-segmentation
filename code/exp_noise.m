% noise experiments

%% dataset setting
brainwebFolder = '/Users/shake/Documents/master/project/MRBrainSeg/dataset/brainweb';
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

% method: BiLSTM/LSTM
numHiddenUnits = 40;
numClasses = 4;
maxEpochs = 1;
MiniBatchSize = 512;
slicNum = 2000;
compactness = 10;

% set save path
modelPath = '/Users/shake/Documents/master/project/MRBrainSeg/fig5';

% train and save models_lstm/bilstm
%     fprintf('training: feature mode = %d, n=%d\n', featureMode, numHiddenUnits);
%     net = trainPixel(t1Train, t2Train, pdTrain, gtTrain, numTrain, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['bilstm_' noiseLevel{level} '.mat'];
%     save(fullfile(modelPath, net_name), 'net');


% train and save models_sliclstm/bilstm


% one image test on s099
% test1_Idx = 153;  % s099
% imagename = 's099';
%     test1_Idx = 64;  % c080
%     imagename = 'c080';
test1_Idx = 99;  % t075
imagename = 't075';    

% bilstm/lstm_s099
featureMode = 4;

load('C:\Users\shake\Desktop\MRBrainSeg\trained model\model_different_methods_brainweb\model_bilstm_f4.mat', 'net');
[test1_acc, test1_predImage, gtimage, predOverlayt1, gtOverlayt1] = testOnePixel(cmap, brainwebFolder, net, test1_Idx, t1Test, t2Test, pdTest, gtTest, featureMode, MiniBatchSize);
fprintf('%f\n', test1_acc);

predname1 = [imagename 'bilstm_pred.bmp'];
predname3 = [imagename 'bilstm_gt.bmp'];
predname2 = [imagename 'bilstm_predoverlay.bmp'];
predname4 = [imagename 'bilstm_gtoverlay.bmp'];
imwrite(test1_predImage,fullfile(modelPath, predname1));
imwrite(predOverlayt1,fullfile(modelPath, predname2));
imwrite(gtOverlayt1,fullfile(modelPath, predname3));
imwrite(gtimage,fullfile(modelPath, predname4))


% slic+lstm/bilstm_s099
featureMode = 4;
load('C:\Users\shake\Desktop\MRBrainSeg\trained model\model_slic_params_brainweb\model_slic_listm_2000_10_6.mat', 'net');
[test1_acc, test1_predImage, gtimage, predOverlayt1, gtOverlayt1] = testOne(cmap, brainwebFolder, net, test1_Idx, t1Test, t2Test, pdTest, gtTest, slicNum, compactness, featureMode, MiniBatchSize);
fprintf('%f\n', test1_acc);

predname1 = [imagename 'slic_lstm_pred.bmp'];
predname3 = [imagename 'slic_lstm_gt.bmp'];
predname2 = [imagename 'slic_lstm_predoverlay.bmp'];
predname4 = [imagename 'slic_lstm_gtoverlay.bmp'];
imwrite(test1_predImage,fullfile(modelPath, predname1));
imwrite(predOverlayt1,fullfile(modelPath, predname2));
imwrite(gtOverlayt1,fullfile(modelPath, predname3));
imwrite(gtimage,fullfile(modelPath, predname4));


% fcm
k = 4;
fcmmodel = 'fcm';
[imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = fcmTest(brainwebFolder, test1_Idx, t1Test, gtTest, k, cmap);
fprintf('fcm(%s):%f/%f\n', imagename, test1_acc, test1_dsc);

predname1 = [imagename '_pred_fcm.bmp'];
predname2 = [imagename '_predoverlay_fcm.bmp'];
imwrite(test1_predImage,fullfile(modelPath, predname1));
imwrite(predOverlayt1,fullfile(modelPath, predname2));

% svm
load('C:\Users\shake\Desktop\MRBrainSeg\trained model\model_different_methods_brainweb\model_svm_10_f2.mat');
[imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = classifierTest(brainwebFolder, model_svm_10_f2, test1_Idx, t1Test, t2Test, pdTest, gtTest, cmap);
fprintf('svm(%s):%f/%f\n', imagename, test1_acc, test1_dsc);

predname1 = [imagename '_svm_pred.bmp'];
predname2 = [imagename '_svm_predoverlay.bmp'];
imwrite(test1_predImage,fullfile(modelPath, predname1));
imwrite(predOverlayt1,fullfile(modelPath, predname2));

% segnet
[imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = segnetTest(brainwebFolder, segnetd2, test1_Idx, t1Test, gtTest, cmap);
fprintf('segnet(%s):%f/%f\n', imagename, test1_acc, test1_dsc);

predname1 = [imagename '_segnet_pred.bmp'];
predname2 = [imagename '_segnet_predoverlay.bmp'];
imwrite(test1_predImage, fullfile(modelPath, predname1));
imwrite(predOverlayt1, fullfile(modelPath, predname2));


%% fcn gray2rgb
folder = 'C:\Users\shake\Desktop\MRBrainSeg\dataset\MRBrainS\t1';
t1 = imageDatastore(folder);
for i=1:174
    [img, info] = readimage(t1, i);
    [rows,cols]=size(img);
    r=double(img);
    g=double(img);
    b=double(img);
    rgb=uint8(cat(3,r,g,b));
    name = info.Filename(end-7:end);
    imwrite(rgb, fullfile('C:\Users\shake\Desktop\MRBrainSeg\dataset\MRBrainS\t1_rgb',name));
    
end



