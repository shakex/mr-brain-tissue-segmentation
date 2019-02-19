function [test_metric, superpixel_metric] = testAll(net, t1Test, t2Test, pdTest, gtTest, numTest, slicNum, compactness, featureMode, MiniBatchSize)

%% test on the whole dataset (brainweb,399=239+160)
test_metric = struct('testTime',0,...
    'classifyAccAll',zeros(numTest, 1),'classifyAcc',zeros(1,2),...
    'AccAll',zeros(numTest, 1),'Acc',zeros(1,2),...
    'JSAll',zeros(numTest, 4),'JS',zeros(4,2),'JS_avg',zeros(1,2),...
    'DSCAll',zeros(numTest, 4),'DSC',zeros(4,2),'DSC_avg',zeros(1,2));
superpixel_metric = struct('numRegions',0,...
    'superpixelAccAll',zeros(numTest, 1),'superpixelAcc',zeros(1,2),...
    'superpixelTime',0);

% var = 0.01;
for test_Idx = 1:numTest
    fprintf('%d/%d \t', test_Idx, numTest);
    [test_t1, testInfo] = readimage(t1Test, test_Idx);
    test_t2 = readimage(t2Test, test_Idx);
    test_pd = readimage(pdTest, test_Idx);
    
    % add nosie (gaussian)
%     test_t1 = imnoise(test_t1,'gaussian',0,var);
%     test_t2 = imnoise(test_t2,'gaussian',0,var);
%     test_pd = imnoise(test_pd,'gaussian',0,var);
    
    startTime = clock;  % start eval method time
    [test_L, test_N] = superpixels(test_t1, slicNum, 'Compactness', compactness);
    endTime_slic = clock;
    [test_L, test_N] = removeBackground(test_t1, test_L, test_N, 0.05);
    test_A = constructAMat(test_L, test_N);
    test_G = graph(test_A);
    
    test_X = createSeq(test_t1, test_t2, test_pd, test_G, test_L, test_N, featureMode);
    test_Y = createLabel(gtTest, test_Idx, test_L, test_N);
    
    test_pred = classify(net, test_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');
    [test_predlbl, test_predImage] = createPredImage(test_pred, test_L, test_N);
    
    endTime = clock;    % end eval method time
    time = etime(endTime, startTime);
    time_slic = etime(endTime_slic, startTime);
    
    % compute metrics
    test_gt = readimage(gtTest, test_Idx);
    [test_lbl, test_lblImage] = createPredImage(test_Y, test_L, test_N);
    
    numRegions = test_N;
    superpixelAcc = sum(sum(test_lbl == double(test_gt)))./numel(test_lbl);
    superpixelTime = time_slic;
    testTime = time;
    classifyAcc = sum(test_pred == categorical(test_Y))./numel(categorical(test_Y));
    Acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);
    JS = jaccard(test_predlbl, double(test_gt));
    DSC = dice(test_predlbl, double(test_gt));
    
    
    superpixel_metric(1).numRegions = superpixel_metric(1).numRegions + numRegions;
    superpixel_metric(1).superpixelAccAll(test_Idx) = superpixelAcc;
    superpixel_metric(1).superpixelTime = superpixel_metric(1).superpixelTime + superpixelTime;
    test_metric(1).testTime = test_metric(1).testTime + testTime;
    test_metric(1).classifyAccAll(test_Idx) = classifyAcc;
    test_metric(1).AccAll(test_Idx) = Acc;
    test_metric(1).JSAll(test_Idx,:) = JS';
    test_metric(1).DSCAll(test_Idx,:) = DSC';
    
    fprintf('%.4f \n', sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl));
end

superpixel_metric(1).numRegions = superpixel_metric(1).numRegions / numTest;
superpixel_metric(1).superpixelAcc(1,1) = mean(superpixel_metric(1).superpixelAccAll);
superpixel_metric(1).superpixelAcc(1,2) = std(superpixel_metric(1).superpixelAccAll);
superpixel_metric(1).superpixelTime = superpixel_metric(1).superpixelTime / numTest;
test_metric(1).testTime = test_metric(1).testTime / numTest;
test_metric(1).classifyAcc(1,1) = mean(test_metric(1).classifyAccAll);
test_metric(1).classifyAcc(1,2) = std(test_metric(1).classifyAccAll);
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


end
