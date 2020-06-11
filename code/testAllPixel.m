function test_metric = testAllPixel(net, t1Test, t2Test, pdTest, gtTest, numTest, featureMode, MiniBatchSize)

%% test on the whole dataset (brainweb,399=239+160)
test_metric = struct('testTime',0,...
    'classifyAccAll',zeros(numTest, 1),'classifyAcc',zeros(1,2),...
    'AccAll',zeros(numTest, 1),'Acc',zeros(1,2),...
    'JSAll',zeros(numTest, 4),'JS',zeros(4,2),'JS_avg',zeros(1,2),...
    'DSCAll',zeros(numTest, 4),'DSC',zeros(4,2),'DSC_avg',zeros(1,2));

d = 0.20;
var = 0.03;
for test_Idx = 1:numTest
    fprintf('%d/%d \t', test_Idx, numTest);
    [test_t1, testInfo] = readimage(t1Test, test_Idx);
    test_t2 = readimage(t2Test, test_Idx);
    test_pd = readimage(pdTest, test_Idx);
    
    % add nosie (sp)
%     test_t1 = imnoise(test_t1,'salt & pepper',d);
%     test_t2 = imnoise(test_t2,'salt & pepper',d);
%     test_pd = imnoise(test_pd,'salt & pepper',d); 
    
    % add nosie (gaussian)
%     test_t1 = imnoise(test_t1,'gaussian',0,var);
%     test_t2 = imnoise(test_t2,'gaussian',0,var);
%     test_pd = imnoise(test_pd,'gaussian',0,var);
    
    startTime = clock;  % start eval method time
    test_X = createPixelSeq(test_t1, test_t2, test_pd, featureMode);
    test_Y = createPixelLabel(gtTest, test_Idx);
    
    test_pred = classify(net, test_X, 'MiniBatchSize', MiniBatchSize, 'SequenceLength', 'longest');
    [test_predlbl, test_predImage] = createPixelPredImage(test_t1, test_pred);
    
    endTime = clock;    % end eval method time  
    time = etime(endTime, startTime);
    
    % compute metrics
    test_gt = readimage(gtTest, test_Idx);
    [test_lbl, test_lblImage] = createPixelPredImage(test_gt, test_Y);
      
    testTime = time;
    classifyAcc = sum(test_pred == categorical(test_Y))./numel(categorical(test_Y));
    Acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);
    JS = jaccard(test_predlbl, double(test_gt));
    DSC = dice(test_predlbl, double(test_gt));
        
    test_metric(1).testTime = test_metric(1).testTime + testTime;
    test_metric(1).classifyAccAll(test_Idx) = classifyAcc;
    test_metric(1).AccAll(test_Idx) = Acc;
    test_metric(1).JSAll(test_Idx,:) = JS';
    test_metric(1).DSCAll(test_Idx,:) = DSC';
    
    fprintf('%.4f \n', sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl));
end


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