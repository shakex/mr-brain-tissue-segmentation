% Experiment-1
% dataset: mrbrains
% slic+lstm,superpixel value exp

%% dataset setting
mrbriansFolder = 'C:\Users\shake\Desktop\MRBrainSeg\dataset\MRBrainS';
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
% sampleIm = readimage(t1, 100);
% samplegt = readimage(gt, 100);
% sampleOverlay = labeloverlay(sampleIm, samplegt, 'ColorMap', cmap, 'Transparency',0.6);
% imshow(sampleOverlay);
% pixelLabelColorbar(cmap, classes);

%% method parameters setting
% superpixel parameters
% slicNum = [500:250:4000];
% compactness = [5 10 20];

% feature extraction mode
featureMode = 5;
numFeature = 3;

% network hyperparameters
numHiddenUnits = 40;
numClasses = 4;
maxEpochs = 20;
MiniBatchSize = 512;

% set save path
modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\model_noise_bilstm_MRbrainS\new';
metricPath = 'C:\Users\shake\Desktop\MRBrainSeg\test results\metric_noise_bilstm_MRbrainS\new';

% train and save models
for var = 0.01:0.02:0.09
for compactness = 10
    for slicNum = 2500
        fprintf('training: compactness=%d, slicNum=%d\n', compactness, slicNum);
        net = train(var, t1Train, t1irTrain, t2flairTrain, gtTrain, numTrain, slicNum, compactness, featureMode, numFeature, maxEpochs, MiniBatchSize, numHiddenUnits, numClasses);
        net_name = ['model_slic_bilstm_f4_g' num2str(var*100) '.mat'];
        save(fullfile(modelPath, net_name), 'net');
    end
end
end

% test on dataset and save results
for var = 0.01:0.02:0.09
for compactness = 10
    for slicNum = 2500
        fprintf('testing: compactness=%d, slicNum=%d\n', compactness, slicNum);
        net_name = ['model_slic_bilstm_f4_g' num2str(var*100) '.mat'];
        load(fullfile(modelPath, net_name), 'net');
        [test_metric, superpixel_metric] = testAll(var, net, t1Test, t1irTest, t2flairTest, gtTest, numTest, slicNum, compactness, featureMode, MiniBatchSize);
        test_metric_name = ['test_metric_slic_bilstm_f4_g' num2str(var*100) '.mat'];
        superpixel_metric_name = ['superpixel_metric_slic_bilstm_f4_g' num2str(var*100) '.mat'];
        save(fullfile(metricPath, test_metric_name), 'test_metric');
        save(fullfile(metricPath, superpixel_metric_name), 'superpixel_metric');
    end
end
end


%% One image test step1 
test1_Idx = 66;
[test1_t1, test1_info] = readimage(t1Test, test1_Idx);

test1_t1ir = readimage(t1irTest, test1_Idx);
test1_t2flair = readimage(t2flairTest, test1_Idx);

[test1_L, test1_N] = superpixels(test1_t1, 2500, 'Compactness', 10);

% display superpixel oversegment images
dispSuperpixel(test1_t1, test1_L, test1_N, true, 4);

[test1_L, test1_N] = removeBackground(test1_t1, test1_L, test1_N, 0.05);
test1_A = constructAMat(test1_L, test1_N);
test1_G = graph(test1_A);

% display superpixel graph
dispSuperpixelGraph(test1_t1, test1_L, test1_N, test1_G, 4);

test1_X = createSeq(test1_t1, test1_t1ir, test1_t2flair, test1_G, test1_L, test1_N, featureMode);
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
figure;imshow(test1_predImageOverlay);
pixelLabelColorbar(cmap, classes);
% imwrite(test1_predImage, 'C:\Users\shake\Desktop\专利附图\7-2.jpg');
% imwrite(label2rgb(double(test1_gt),cmap), 'C:\Users\shake\Desktop\专利附图\7-1.jpg');

% Compare with ground truth 
figure;imshowpair(uint8(test1_predlbl), uint8(test1_gt));

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