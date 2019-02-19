% ComparsionExp on MRBrainS
mrbrainsFolder = 'C:\Users\shake\Desktop\MRBrainSeg\dataset\MRBrainS';
classes = [
"BG"
"CSF"
"GM"
"WM"
];
cmap = brainwebColorMap;
gtIDs = [1 2 3 4];

t1Dir = fullfile(mrbrainsFolder, 't1_rgb');
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

% method: BiLSTM/LSTM
numHiddenUnits = 40;

numClasses = 4;
maxEpochs = 30;
MiniBatchSize = 512;
slicNum = 2000;
compactness = 10;

% set save path
modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\comparsionExp';
imagePath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\comparsionExp\visualization';

%% Training
% train and save models_lstm/bilstm
% featureMode = 4;
% maxEpochs = 1;
% rnnmodel = {'bilstm','lstm'};
% for modelIdx = 1:2
%     fprintf('training %s\n', rnnmodel{modelIdx});
%     net = trainPixel(t1Train, t1irTrain, t2flairTrain, gtTrain, numTrain, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = [rnnmodel{modelIdx} '.mat'];
%     save(fullfile(modelPath, net_name), 'net');
% end

% train and save models_slic_lstm/bilstm
% featureMode = 4;
% maxEpochs = 30;
% rnnmodel = {'bilstm','lstm'};
% for modelIdx = 1:2
%     fprintf('training slic+%s\n', rnnmodel{modelIdx});
%     net = train(t1Train, t1irTrain, t2flairTrain, gtTrain, numTrain, slicNum, compactness, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['slic_' rnnmodel{modelIdx} '.mat'];
%     save(fullfile(modelPath,net_name), 'net');
% end
% 
% % extract features fed into classifiers {'knn','svm','tree'}
% pixelTrain = extractFeaturesTrain(t1Train, t1irTrain, t2flairTrain, gtTrain);
% save(fullfile(modelPath, 'extractedfeatures', '1extractedfeatures.mat'), 'pixelTrain');

% train and save fcn/segnet/unet
unetd2 = trainCNN(t1Train, gtTrain);

% save(fullfile(modelPath, 'model_segnetd2_150.mat'), 'segnetd2');
% save(fullfile(modelPath, 'model_segnetd3_150.mat'), 'segnetd3');
save(fullfile(modelPath, 'unet2.mat'), 'unetd2');
% save(fullfile(modelPath, 'segnet.mat'), 'segnetd2');
% save(fullfile(modelPath, 'fcn8s.mat'), 'fcn8s');


%% Testing
% test models_lstm/bilstm
featureMode = 4;
rnnmodel = {'bilstm','lstm'};
for modelIdx = 1:2
    net_name = [rnnmodel{modelIdx} '.mat'];
    load(fullfile(modelPath, net_name), 'net');
    accAll = 0;
    dscAll = 0;
    for test_Idx = 1:numTest
        [imagename, test1_acc, test1_dsc, test1_predImage, gtimage, predOverlayt1, gtOverlayt1] = testOnePixel(cmap, mrbrainsFolder, net, test_Idx, t1Test, t1irTest, t2flairTest, gtTest, featureMode, MiniBatchSize);
        fprintf('%s(%s):%f/%f\n', rnnmodel{modelIdx}, imagename, test1_acc, test1_dsc);
        accAll = accAll + test1_acc;
        dscAll = dscAll + test1_dsc;
        
        predname1 = [imagename '_' rnnmodel{modelIdx} '_pred.bmp'];
        predname2 = [imagename '_' rnnmodel{modelIdx} '_predoverlay.bmp'];
        imwrite(test1_predImage,fullfile(imagePath, predname1));
        imwrite(predOverlayt1,fullfile(imagePath, predname2));
    end
    accAll = accAll / numTest;
    dscAll = dscAll / numTest;
    fprintf('Segmentation Results %s: ACC=%f, DSC=%f\n', rnnmodel{modelIdx}, accAll, dscAll);
end

%% test models_slic_lstm/bilstm
featureMode = 4;
rnnmodel = {'bilstm','lstm'};
for modelIdx = 1:2
    net_name = ['slic_' rnnmodel{modelIdx} '.mat'];
    load(fullfile(modelPath, net_name), 'net');
    accAll = 0;
    dscAll = 0;
    for test_Idx = 1:numTest
        [imagename, test1_acc, test1_dsc, test1_predImage, gtimage, predOverlayt1, gtOverlayt1] = testOne(cmap, mrbrainsFolder, net, test_Idx, t1Test, t1irTest, t2flairTest, gtTest, slicNum, compactness, featureMode, MiniBatchSize);
        fprintf('slic+%s(%s):%f/%f\n', rnnmodel{modelIdx}, imagename, test1_acc, test1_dsc);
        accAll = accAll + test1_acc;
        dscAll = dscAll + test1_dsc;
        
        predname1 = [imagename '_slic_' rnnmodel{modelIdx} '_pred.bmp'];
        predname2 = [imagename '_slic_' rnnmodel{modelIdx} '_predoverlay.bmp'];
        imwrite(test1_predImage,fullfile(imagePath, predname1));
        imwrite(predOverlayt1,fullfile(imagePath, predname2));
    end
    accAll = accAll / numTest;
    dscAll = dscAll / numTest;
    fprintf('Segmentation Results SLIC+%s: ACC=%f, DSC=%f\n', rnnmodel{modelIdx}, accAll, dscAll);
end

%%
% test kmeans
k = 4;
accAll = 0;
dscAll = 0;
for test_Idx = 1:numTest
    [imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = kmeansTest(mrbrainsFolder, test_Idx, t1Test, gtTest, k, cmap);
    fprintf('kmeans(%s):%f/%f\n', imagename, test1_acc, test1_dsc);
    accAll = accAll + test1_acc;
    dscAll = dscAll + test1_dsc;
    
    predname1 = [imagename '_kmeans_pred.bmp'];
    predname2 = [imagename '_kmeans_predoverlay.bmp'];
    imwrite(test1_predImage,fullfile(imagePath, predname1));
    imwrite(predOverlayt1,fullfile(imagePath, predname2));
end
accAll = accAll / numTest;
dscAll = dscAll / numTest;
fprintf('Segmentation Results K-Means: ACC=%f, DSC=%f\n', accAll, dscAll);

% test fcm
k = 4;
accAll = 0;
dscAll = 0;
for test_Idx = 1:numTest
    [imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = fcmTest(mrbrainsFolder, test_Idx, t1Test, gtTest, k, cmap);
    fprintf('fcm(%s):%f/%f\n', imagename, test1_acc, test1_dsc);
    accAll = accAll + test1_acc;
    dscAll = dscAll + test1_dsc;
    
    predname1 = [imagename '_fcm_pred.bmp'];
    predname2 = [imagename '_fcm_predoverlay.bmp'];
    imwrite(test1_predImage,fullfile(imagePath, predname1));
    imwrite(predOverlayt1,fullfile(imagePath, predname2));
end
accAll = accAll / numTest;
dscAll = dscAll / numTest;
fprintf('Segmentation Results FCM: ACC=%f, DSC=%f\n', accAll, dscAll);

%% 
% test knn/svm/tree
classifier = {'knn','svm','tree'};
accAll = 0;
dscAll = 0;
for modelIdx = 1:3
    for test_Idx = 1:numTest
        model_name = [classifier{modelIdx} '.mat'];
        load(fullfile(modelPath, model_name), 'trainedclassifier');
        [imagename, test1_acc, test1_dsc, test1_predImage, predOverlayt1] = classifierTest(mrbrainsFolder, trainedclassifier, test_Idx, t1Test, t1irTest, t2flairTest, gtTest, cmap);
        fprintf('%s(%s):%f/%f\n', classifier{modelIdx}, imagename, test1_acc, test1_dsc);
        accAll = accAll + test1_acc;
        dscAll = dscAll + test1_dsc;

        predname1 = [imagename '_' classifier{modelIdx} '_pred.bmp'];
        predname2 = [imagename '_' classifier{modelIdx} 'predoverlay.bmp'];
        imwrite(test1_predImage,fullfile(imagePath, predname1));
        imwrite(predOverlayt1,fullfile(imagePath, predname2));
    end
    accAll = accAll / numTest;
    dscAll = dscAll / numTest;
    fprintf('Segmentation Results %s: ACC=%f, DSC=%f\n', classifier{modelIdx}, accAll, dscAll);
end


%% test fcn/unet/segnet
test_metric = testAllCNN_mrbrains(fcn8s, t1Test, gtTest, numTest);











