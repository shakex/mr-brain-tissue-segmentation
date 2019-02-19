% noise experiments

%% dataset setting
noiseLevel = {'pn1', 'pn3', 'pn5', 'pn7', 'pn9'};
for level = 1:5
    brainwebFolder = strcat('C:\Users\shake\Desktop\MRBrainSeg\dataset\brainweb\noise\',noiseLevel{level});
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
    modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\noise';

    % train and save models_lstm/bilstm
%     fprintf('training: feature mode = %d, n=%d\n', featureMode, numHiddenUnits);
%     net = trainPixel(t1Train, t2Train, pdTrain, gtTrain, numTrain, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
%     net_name = ['bilstm_' noiseLevel{level} '.mat'];
%     save(fullfile(modelPath, net_name), 'net');
    
    % train and save models_sliclstm/bilstm


    % one image test on s099
%     test1_Idx = 153;  % s099
%     test1_Idx = 64;  % c080
%     imagename = 'c080';
    test1_Idx = 99;  % t075
    imagename = 't075';    

    % bilstm/lstm_s099
    featureMode = 4;
    rnnmodel = {'bilstm','lstm'};
    for modelIdx = 1:2
        net_name = [rnnmodel{modelIdx} '_' noiseLevel{level} '.mat'];
        load(fullfile(modelPath, net_name), 'net');
        [test1_acc, test1_predImage, predOverlayt1] = testOnePixel(brainwebFolder, net, test1_Idx, t1Test, t2Test, pdTest, gtTest, featureMode, MiniBatchSize);
        fprintf('%s(%s):%f\n', rnnmodel{modelIdx}, noiseLevel{level}, test1_acc);

        predname1 = [imagename '_pred_' rnnmodel{modelIdx} '_' noiseLevel{level} '.bmp'];
        predname2 = [imagename '_predoverlay_' rnnmodel{modelIdx} '_' noiseLevel{level} '.bmp'];
        imwrite(test1_predImage,fullfile(modelPath, predname1));
        imwrite(predOverlayt1,fullfile(modelPath, predname2));
    end

    % slic+lstm/bilstm_s099
    featureMode = 6;
    rnnmodel = {'bilstm','lstm'};
    for modelIdx = 1:2
        net_name = ['slic_' rnnmodel{modelIdx} '_' noiseLevel{level} '.mat'];
        load(fullfile(modelPath, net_name), 'net');
        [test1_acc, test1_predImage, predOverlayt1] = testOne(brainwebFolder, net, test1_Idx, t1Test, t2Test, pdTest, gtTest, slicNum, compactness, featureMode, MiniBatchSize);
        fprintf('slic_%s(%s):%f\n', rnnmodel{modelIdx}, noiseLevel{level}, test1_acc);

        predname1 = [imagename '_pred_slic' rnnmodel{modelIdx} '_' noiseLevel{level} '.bmp'];
        predname2 = [imagename '_predoverlay_slic' rnnmodel{modelIdx} '_' noiseLevel{level} '.bmp'];
        imwrite(test1_predImage,fullfile(modelPath, predname1));
        imwrite(predOverlayt1,fullfile(modelPath, predname2));
    end
    
    % kmeans
    k = 4;
    kmeansmodel = 'kmeans';
    [test1_acc, test1_predImage, predOverlayt1] = kmeansTest(brainwebFolder, test1_Idx, t1Test, gtTest, k, cmap);
    fprintf('%s(%s):%f\n', kmeansmodel, noiseLevel{level}, test1_acc);
    
    predname1 = [imagename '_pred_' kmeansmodel '_' noiseLevel{level} '.bmp'];
    predname2 = [imagename '_predoverlay_' kmeansmodel '_' noiseLevel{level} '.bmp'];
    imwrite(test1_predImage,fullfile(modelPath, predname1));
    imwrite(predOverlayt1,fullfile(modelPath, predname2));
    
    % fcm
    k = 4;
    fcmmodel = 'fcm';
    [test1_acc, test1_predImage, predOverlayt1] = fcmTest(brainwebFolder, test1_Idx, t1Test, gtTest, k, cmap);
    fprintf('%s(%s):%f\n', fcmmodel, noiseLevel{level}, test1_acc);
    
    predname1 = [imagename '_pred_' fcmmodel '_' noiseLevel{level} '.bmp'];
    predname2 = [imagename '_predoverlay_' fcmmodel '_' noiseLevel{level} '.bmp'];
    imwrite(test1_predImage,fullfile(modelPath, predname1));
    imwrite(predOverlayt1,fullfile(modelPath, predname2));
    
%     SVM/KNN/DecisionTree/Linear discriminant
%     extract features
%     pixelTrain = extractFeaturesTrain(t1Train, t2Train, pdTrain, gtTrain);
%     save(['10features_' noiseLevel{level}], 'pixelTrain');
    
    % test on s099
    classifier = {'knn','svm','tree'};
    for modelIdx = 1:3
        model_name = [classifier{modelIdx} '_' noiseLevel{level} '.mat'];
        load(fullfile(modelPath, model_name), 'trainedclassifier');
        [test1_acc, test1_predImage, predOverlayt1] = classifierTest(brainwebFolder, trainedclassifier, test1_Idx, t1Test, t2Test, pdTest, gtTest, cmap);
        fprintf('%s(%s):%f\n', classifier{modelIdx}, noiseLevel{level}, test1_acc);

        predname1 = [imagename '_pred_' classifier{modelIdx} '_' noiseLevel{level} '.bmp'];
        predname2 = [imagename '_predoverlay_' classifier{modelIdx} '_' noiseLevel{level} '.bmp'];
        imwrite(test1_predImage,fullfile(modelPath, predname1));
        imwrite(predOverlayt1,fullfile(modelPath, predname2));
    end
    
end

%% SVM/KNN/DecisionTree/Linear discriminant Training
load(fullfile(modelPath,'extractedfeatures','10features_pn9.mat'), 'pixelTrain');


%% K-Means
k = 4;
time = 0;
accAll = 0;
result = 0;
for iter = 1:10
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

        accAll = accAll + sum(sum(predlbl == double(lbl)))./numel(predlbl);

    end
    accAll = accAll / numTest;
    result = result + accAll;
end

result = result/10



%% one image test



