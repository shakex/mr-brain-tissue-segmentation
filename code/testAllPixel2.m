function test_metric = testAllPixel2(net, t1Test, t2Test, pdTest, gtTest, numTest, featureMode)

%% test on the whole dataset (brainweb,399=239+160)
test_metric = struct('testTime',0,...
    'AccAll',zeros(numTest, 1), 'Acc', zeros(1,2),...
    'JSAll',zeros(numTest, 4), 'JS', zeros(4,2), 'JS_avg', zeros(1,2),...
    'DSCAll',zeros(numTest, 4), 'DSC', zeros(4,2), 'DSC_avg', zeros(1,2));

for test_Idx = 1:numTest
    fprintf('FeatureMode = %d: %d/%d\n', featureMode, test_Idx, numTest);
    test_t1 = readimage(t1Test, test_Idx);
    test_t2 = readimage(t2Test, test_Idx);
    test_pd = readimage(pdTest, test_Idx);

    startTime = clock; % start eval method time
    [row, col] = size(test_t1);
    Array = zeros(numel(test_t1), 3);
    for i=1:row
        for j=1:col
            Array((i - 1) * col + j, 1) = test_t1(i, j); 
            Array((i - 1) * col + j, 2) = test_t2(i, j); 
            Array((i - 1) * col + j, 3) = test_pd(i, j);
        end
    end
    test_X = array2table(Array);
    test_Y = createPixelLabel(gtTest, test_Idx);
    test_pred = net.predictFcn(test_X);
    [test_predlbl, test_predImage] = createPixelPredImage(test_t1, test_pred);
    
    endTime = clock; % end eval method time
    time = etime(endTime, startTime);
    
    % compute metrics
    test_gt = readimage(gtTest, test_Idx);
    [test_lbl, test_lblImage] = createPixelPredImage(test_gt, test_Y);
    
    testTime = time;
    Acc = sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl);
    JS = jaccard(test_predlbl, double(test_gt));
    DSC = dice(test_predlbl, double(test_gt));
    
    test_metric(1).testTime = test_metric(1).testTime + testTime;
    test_metric(1).AccAll(test_Idx) = Acc;
    test_metric(1).JSAll(test_Idx,:) = JS';
    test_metric(1).DSCAll(test_Idx,:) = DSC';
    
    fprintf('%.4f \n', sum(sum(test_predlbl == double(test_gt)))./numel(test_predlbl));

end

test_metric(1).testTime = test_metric(1).testTime / numTest;
test_metric(1).Acc(1,1) = mean(test_metric(1).AccAll);
test_metric(1).Acc(1,2) = std(test_metric(1).AccAll);
test_metric(1).JS(:,1) = (mean(test_metric(1).JSAll,1))';
test_metric(1).JS(:,2) = (std(test_metric(1).JSAll,1))';
test_metric(1).JS_avg(1,1) = mean(test_metric(1).JS(:,1));
test_metric(1).JS_avg(1,2) = std(test_metric(1).JS(:,2));
test_metric(1).DSC(:,1) = (mean(test_metric(1).DSCAll, 1))';
test_metric(1).DSC(:,2) = (std(test_metric(1).DSCAll, 1))';
test_metric(1).DSC_avg(1,1) = mean(test_metric(1).DSC(:,1));
test_metric(1).DSC_avg(1,2) = std(test_metric(1).DSC(:,2));
