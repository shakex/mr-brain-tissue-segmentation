function net = train(t1Train, t2Train, pdTrain, gtTrain, numTrain, slicNum, compactness, featureMode, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses)

%% superpixel
% for trainIdx = 1:numTrain
XTrain = {};
YTrain = {};

for trainIdx = 1:numTrain
    imt1 = readimage(t1Train, trainIdx);
    imt2 = readimage(t2Train, trainIdx);
    impd = readimage(pdTrain, trainIdx);
    
    % add nosie (gaussian)
%     imt1 = imnoise(imt1,'gaussian',0,var);
%     imt2 = imnoise(imt2,'gaussian',0,var);
%     impd = imnoise(impd,'gaussian',0,var);
    
    [L, N] = superpixels(imt1, slicNum, 'Compactness', compactness);    % slic
    
    % display superpixel oversegment images
%     dispSuperpixel(imt1, L, N, true, 4);
    
    [L, N] = removeBackground(imt1, L, N, 0.05);  % Remove background
    
    A = constructAMat(L, N);
    G = graph(A);
    
    % display superpixel graph
%     dispSuperpixelGraph(imt1, L, N, G, 4);
    
    % display subgraph
%     dispSubGraph(imt1, L, N, G, 501, 550, 4);


    trainSeq = createSeq(imt1, imt2, impd, G, L, N, featureMode);   % extract features

    gtTrainClass = createLabel(gtTrain, trainIdx, L, N);    % prepare gt
    
    XTrain = [XTrain;trainSeq];
    YTrain = [YTrain;categorical(gtTrainClass)];
end

%% training configuration
if featureMode == 1
    numFeature = 1; % pixel gray
elseif featureMode == 2
    numFeature = 3; % modality
elseif featureMode == 3
    numFeature = 1; % neighbors
elseif featureMode == 4
    numFeature = 3; % modality&neighbors
end

layers = [ ...
    sequenceInputLayer(numFeature)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',MiniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

options2 = trainingOptions('adam',...
    'GradientThreshold',1, ...
    'GradientDecayFactor',0.95,...
    'SquaredGradientDecayFactor',0.999,...
    'InitialLearnRate',0.005,...
    'LearnRateSchedule','piecewise',... 
    'LearnRateDropFactor',0.8,... 
    'LearnRateDropPeriod',5,... 
    'MaxEpochs',maxEpochs,... 
    'MiniBatchSize',MiniBatchSize,...
    'Verbose',false,...
    'Plots','training-progress');

%% start training
net = trainNetwork(XTrain,YTrain,layers,options2);

end