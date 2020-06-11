% plot noise test results
test_Idx = 48;
MiniBatchSize = 512;
modelPath = 'C:\Users\shake\Desktop\MRBrainSeg\trained model\model_noise_bilstm_MRbrainS';
kmeansmodelPath = 'C:\Users\shake\Desktop\MRBrainSeg\test results\metric_noise_kmeans_MRbrianS';

test_gt = readimage(gtTest, test_Idx);
figure;imshow(label2rgb(double(test_gt),cmap));title('ground truth');

%% slic+bilstm(f6)
featureMode = 6;
figure;
% noise = 0
test_t1 = readimage(t1Test, test_Idx);
test_t1ir = readimage(t1irTest, test_Idx);
test_t2flair = readimage(t2flairTest, test_Idx);
[test_L, test_N] = superpixels(test_t1, slicNum, 'Compactness', compactness);
[test_L, test_N] = removeBackground(test_t1, test_L, test_N, 0.05);
test_sp = dispSuperpixel(test_t1, test_L, test_N, 1, 1);
test_A = constructAMat(test_L, test_N);
test_G = graph(test_A);
test_X = createSeq(test_t1, test_t1ir, test_t2flair, test_G, test_L, test_N, featureMode);
net_name = 'model_slic_bilstm_g0.mat';
load(fullfile(modelPath, net_name), 'net');
test_pred = classify(net, test_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');
[test_predlbl, test_predImage] = creatPredImage(test_pred, test_L, test_N);
acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);
subplot(4,6,1);imshow(test_t1);title('Gaussian m=0 var=0');
subplot(4,6,7);imshow(test_sp);title('slic');
subplot(4,6,13);imshow(test_predImage);title('Pred');
subplot(4,6,19);imshowpair(uint8(test_predlbl), uint8(test_gt));title(['Difference (acc=' num2str(acc) ')']);

for var = 0.01:0.02:0.09
    test_t1 = readimage(t1Test, test_Idx);
    test_t1ir = readimage(t1irTest, test_Idx);
    test_t2flair = readimage(t2flairTest, test_Idx);
    
    test_t1 = imnoise(test_t1,'gaussian',0,var);
    test_t1ir = imnoise(test_t1ir,'gaussian',0,var);
    test_t2flair = imnoise(test_t2flair,'gaussian',0,var);
     
    [test_L, test_N] = superpixels(test_t1, slicNum, 'Compactness', compactness);
    [test_L, test_N] = removeBackground(test_t1, test_L, test_N, 0.05);
    test_sp = dispSuperpixel(test_t1, test_L, test_N, 1, 1);
    test_A = constructAMat(test_L, test_N);
    test_G = graph(test_A);
    test_X = createSeq(test_t1, test_t1ir, test_t2flair, test_G, test_L, test_N, featureMode);
    net_name = ['model_slic_bilstm_g' num2str(var*100) '.mat'];
    load(fullfile(modelPath, net_name), 'net');
    test_pred = classify(net, test_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');
    [test_predlbl, test_predImage] = creatPredImage(test_pred, test_L, test_N);
    
    acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);
   
    % plot img,pred,difference
    subplot(4,6,1+(var+0.01)/0.02);imshow(test_t1);title(['Gaussian ' 'm=0 ' 'var=' num2str(var)])
    subplot(4,6,7+(var+0.01)/0.02);imshow(test_sp);title('slic');
    subplot(4,6,13+(var+0.01)/0.02);imshow(test_predImage);title('Pred');
    subplot(4,6,19+(var+0.01)/0.02);imshowpair(uint8(test_predlbl), uint8(test_gt));title(['Difference (acc=' num2str(acc) ')']);

end
%% Bilstm(f4)
featureMode = 4;
figure;
% noise = 0
test_t1 = readimage(t1Test, test_Idx);
test_t1ir = readimage(t1irTest, test_Idx);
test_t2flair = readimage(t2flairTest, test_Idx);
test_X = createPixelSeq(test_t1, test_t1ir, test_t2flair, featureMode);
test_Y = createPixelLabel(gtTest, test_Idx);
net_name = 'model_bilstm_g0.mat';
load(fullfile(modelPath, net_name), 'net');
test_pred = classify(net, test_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');
[test_predlbl, test_predImage] = createPixelPredImage(test_t1, test_pred);
acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);
subplot(3,6,1);imshow(test_t1);title('Gaussian m=0 var=0');
subplot(3,6,7);imshow(test_predImage);title('Pred');
subplot(3,6,13);imshowpair(uint8(test_predlbl), uint8(test_gt));title(['Difference (acc=' num2str(acc) ')']);

for var = 0.01:0.02:0.09
    test_t1 = readimage(t1Test, test_Idx);
    test_t1ir = readimage(t1irTest, test_Idx);
    test_t2flair = readimage(t2flairTest, test_Idx);
    
    test_t1 = imnoise(test_t1,'gaussian',0,var);
    test_t1ir = imnoise(test_t1ir,'gaussian',0,var);
    test_t2flair = imnoise(test_t2flair,'gaussian',0,var);
    
    test_X = createPixelSeq(test_t1, test_t1ir, test_t2flair, featureMode);
    test_Y = createPixelLabel(gtTest, test_Idx);
    
    net_name = ['model_bilstm_g' num2str(var*100) '.mat'];
    load(fullfile(modelPath, net_name), 'net');
    
    test_pred = classify(net, test_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');
    [test_predlbl, test_predImage] = createPixelPredImage(test_t1, test_pred);
    
    acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);

    % plot img,pred,difference
    subplot(3,6,1+(var+0.01)/0.02);imshow(test_t1);title(['Gaussian ' 'm=0 ' 'var=' num2str(var)])
    subplot(3,6,7+(var+0.01)/0.02);imshow(test_predImage);title('Pred');
    subplot(3,6,13+(var+0.01)/0.02);imshowpair(uint8(test_predlbl), uint8(test_gt));title(['Difference (acc=' num2str(acc) ')']);
end


%% k-means
k = 4;
% gaussian noise/test metric
for var=0.01:0.02:0.09
    time = 0;
    Acc = 0;
    test_metric = struct('testTime',0,...
    'AccAll',zeros(numTest, 1),'Acc',zeros(1,2),...
    'JSAll',zeros(numTest, 4),'JS',zeros(4,2),'JS_avg',zeros(1,2),...
    'DSCAll',zeros(numTest, 4),'DSC',zeros(4,2),'DSC_avg',zeros(1,2));
    for testIdx = 1:numTest
        startTime = clock;
        [test_t1,info] = readimage(t1Test,testIdx);
        test_t1 = imnoise(test_t1,'gaussian',0,var);
        [row, col] = size(test_t1);

        X = zeros(numel(test_t1), 1);
        for j = 1:col
            for i = 1:row
               X((j - 1) * row + i,1) = test_t1(i,j);
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

        test_predlbl = reshape(kmeansIdx,[row,col]);
        test_gt = readimage(gtTest,testIdx);
        test_predImage = label2rgb(test_predlbl, cmap);

        endTime = clock;
        testTime = etime(endTime,startTime);

        Acc = Acc + sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);
        JS = jaccard(test_predlbl, double(test_gt));
        DSC = dice(test_predlbl, double(test_gt));
        
        test_metric(1).testTime = test_metric(1).testTime + testTime;
        test_metric(1).AccAll(test_Idx) = Acc;
        test_metric(1).JSAll(test_Idx,:) = JS';
        test_metric(1).DSCAll(test_Idx,:) = DSC';

    end
    test_metric(1).testTime = test_metric(1).testTime / numTest;
    test_metric(1).Acc(1,1) = mean(test_metric(1).AccAll);
    test_metric(1).Acc(1,2) = std(test_metric(1).AccAll);
    test_metric(1).JS(:,1) = (mean(test_metric(1).JSAll,1))';
    test_metric(1).JS(:,2) = (std(test_metric(1).JSAll,1))';
    test_metric(1).JS_avg(1,1) = mean(test_metric(1).JS(:,1));
    test_metric(1).JS_avg(1,2) = std(test_metric(1).JS(:,2));
    test_metric(1).DSC(:,1) = (mean(test_metric(1).DSCAll,1))';
    test_metric(1).DSC(:,2) = (std(test_metric(1).DSCAll,1))';
    test_metric(1).DSC_avg(1,1) = mean(test_metric(1).DSC(:,1));
    test_metric(1).DSC_avg(1,2) = std(test_metric(1).DSC(:,2));
    
    % save test metrics
    test_metric_name = ['test_metric_kmeans_g' num2str(var*100) '.mat'];
    save(fullfile(kmeansmodelPath, test_metric_name), 'test_metric');
end

%% kmeans gaussian noise/one test plot
figure;
% noise = 0
test_t1 = readimage(t1Test, test_Idx);
[row, col] = size(test_t1);
X = zeros(numel(test_t1), 1);
for j = 1:col
    for i = 1:row
       X((j - 1) * row + i,1) = test_t1(i,j);
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
test_predlbl = reshape(kmeansIdx,[row,col]);
test_gt = readimage(gtTest,testIdx);
test_predImage = label2rgb(test_predlbl, cmap);

acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);
subplot(3,6,1);imshow(test_t1);title('Gaussian m=0 var=0');
subplot(3,6,7);imshow(test_predImage);title('Pred');
subplot(3,6,13);imshowpair(uint8(test_predlbl), uint8(test_gt));title(['Difference (acc=' num2str(acc) ')']);

for var=0.01:0.02:0.09
    test_t1 = readimage(t1Test, test_Idx);    
    test_t1 = imnoise(test_t1,'gaussian',0,var);
    
    [row, col] = size(test_t1);
    X = zeros(numel(test_t1), 1);
    for j = 1:col
        for i = 1:row
           X((j - 1) * row + i,1) = test_t1(i,j);
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
    test_predlbl = reshape(kmeansIdx,[row,col]);
    test_gt = readimage(gtTest,testIdx);
    test_predImage = label2rgb(test_predlbl, cmap);
    
    acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);

    % plot img,pred,difference
    subplot(3,6,1+(var+0.01)/0.02);imshow(test_t1);title(['Gaussian ' 'm=0 ' 'var=' num2str(var)])
    subplot(3,6,7+(var+0.01)/0.02);imshow(test_predImage);title('Pred');
    subplot(3,6,13+(var+0.01)/0.02);imshowpair(uint8(test_predlbl), uint8(test_gt));title(['Difference (acc=' num2str(acc) ')']);
end







