%% dataset setting
noiseLevel = {'g1', 'g3', 'g5', 'g7', 'g9'};
for var = 0.05
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
    
    % method: BiLSTM/LSTM
    numHiddenUnits = 40;
    
    numClasses = 4;
    maxEpochs = 30;
    MiniBatchSize = 512;
    slicNum = 2000;
    compactness = 10;

    % set save path
    modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\noise_MRbrainS';
    
     % train and save models_lstm/bilstm
%     featureMode = 4;
%     fprintf('training: feature mode = %d, n=%d\n', featureMode, numHiddenUnits);
%     net = trainPixel(var, t1Train, t1irTrain, t2flairTrain, gtTrain, numTrain, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['lstm_g' num2str(var*100) '.mat'];
%     save(fullfile(modelPath, net_name), 'net');
    
    % train and save models_sliclstm/bilstm
%     featureMode = 6;
%     numFeature = 3;
%     fprintf('training: feature mode = %d, n=%d\n', featureMode, numHiddenUnits);
%     net = train(var, t1Train, t1irTrain, t2flairTrain, gtTrain, numTrain, slicNum, compactness, featureMode, numFeature, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['slic_lstm_g' num2str(var*100) '.mat'];
%     save(fullfile(modelPath, net_name), 'net');
%     
    % one image test
%     test1_Idx = 8;  % 2t25
%     imagename = '2t25';
    test1_Idx = 26;  % 3t33
    imagename = '3t33';
%     test1_Idx = 99;  % t075
%     imagename = 't075';  
    
    % slic+bilstm/lstm
    featureMode = 6;
    rnnmodel = {'bilstm','lstm'};
    for modelIdx = 2
        net_name = ['slic_' rnnmodel{modelIdx} '_g' num2str(var*100) '.mat'];
        load(fullfile(modelPath, net_name), 'net');
        [test1_acc, test1_predImage, predOverlayt1] = testOne(var, mrbrainsFolder, net, test1_Idx, t1Test, t1irTest, t2flairTest, gtTest, slicNum, compactness, featureMode, MiniBatchSize);
        fprintf('slic_%s(g%s):%f\n', rnnmodel{modelIdx}, num2str(var*100), test1_acc);

        predname1 = [imagename '_pred_slic' rnnmodel{modelIdx} '_g' num2str(var*100) '.bmp'];
        predname2 = [imagename '_predoverlay_slic' rnnmodel{modelIdx} '_g' num2str(var*100) '.bmp'];
        imwrite(test1_predImage,fullfile(modelPath, predname1));
        imwrite(predOverlayt1,fullfile(modelPath, predname2));
    end
%     
%     % bilstm/lstm
%     featureMode = 4;
%     rnnmodel = {'bilstm','lstm'};
%     for modelIdx = 1:2
%         net_name = [rnnmodel{modelIdx} '_g' num2str(var*100) '.mat'];
%         load(fullfile(modelPath, net_name), 'net');
%         [test1_acc, test1_predImage, predOverlayt1] = testOnePixel(var, mrbrainsFolder, net, test1_Idx, t1Test, t1irTest, t2flairTest, gtTest, featureMode, MiniBatchSize);
%         fprintf('%s(g%s):%f\n', rnnmodel{modelIdx}, num2str(var*100), test1_acc);
% 
%         predname1 = [imagename '_pred_' rnnmodel{modelIdx} '_g' num2str(var*100) '.bmp'];
%         predname2 = [imagename '_predoverlay_' rnnmodel{modelIdx} '_g' num2str(var*100) '.bmp'];
%         imwrite(test1_predImage,fullfile(modelPath, predname1));
%         imwrite(predOverlayt1,fullfile(modelPath, predname2));
%     end
% 
%     % kmeans
%     k = 4;
%     kmeansmodel = 'kmeans';
%     [test1_acc, test1_predImage, predOverlayt1] = kmeansTest(var, mrbrainsFolder, test1_Idx, t1Test, gtTest, k, cmap);
%     fprintf('%s(g%s):%f\n', kmeansmodel, num2str(var*100), test1_acc);
%     
%     predname1 = [imagename '_pred_' kmeansmodel '_g' num2str(var*100) '.bmp'];
%     predname2 = [imagename '_predoverlay_' kmeansmodel '_g' num2str(var*100) '.bmp'];
%     imwrite(test1_predImage,fullfile(modelPath, predname1));
%     imwrite(predOverlayt1,fullfile(modelPath, predname2));
%     
%      % fcm
%     k = 4;
%     fcmmodel = 'fcm';
%     [test1_acc, test1_predImage, predOverlayt1] = fcmTest(var, mrbrainsFolder, test1_Idx, t1Test, gtTest, k, cmap);
%     fprintf('%s(g%s):%f\n', fcmmodel, num2str(var*100), test1_acc);
%     
%     predname1 = [imagename '_pred_' fcmmodel '_g' num2str(var*100) '.bmp'];
%     predname2 = [imagename '_predoverlay_' fcmmodel '_g' num2str(var*100) '.bmp'];
%     imwrite(test1_predImage,fullfile(modelPath, predname1));
%     imwrite(predOverlayt1,fullfile(modelPath, predname2));
%     
% %     SVM/KNN/DecisionTree extract features
% %     pixelTrain = extractFeaturesTrain(var, t1Train, t1irTrain, t2flairTrain, gtTrain);
% %     save(['1features_g' num2str(var*100)], 'pixelTrain');
%     
%     % SVM/KNN/DecisionTree test one
%     classifier = {'knn','svm','tree'};
%     for modelIdx = 1:3
%         model_name = [classifier{modelIdx} '_g' num2str(var*100) '.mat'];
%         load(fullfile(modelPath, model_name), 'trainedclassifier');
%         [test1_acc, test1_predImage, predOverlayt1] = classifierTest(var, mrbrainsFolder, trainedclassifier, test1_Idx, t1Test, t1irTest, t2flairTest, gtTest, cmap);
%         fprintf('%s(g%s):%f\n', classifier{modelIdx}, num2str(var*100), test1_acc);
% 
%         predname1 = [imagename '_pred_' classifier{modelIdx} '_g' num2str(var*100) '.bmp'];
%         predname2 = [imagename '_predoverlay_' classifier{modelIdx} '_g' num2str(var*100) '.bmp'];
%         imwrite(test1_predImage,fullfile(modelPath, predname1));
%         imwrite(predOverlayt1,fullfile(modelPath, predname2));
%     end

end

%% SVM/KNN/DecisionTree/Linear discriminant Training
load(fullfile(modelPath,'extractedfeatures','1features_g5.mat'), 'pixelTrain');

    