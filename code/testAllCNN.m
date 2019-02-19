function test_metric = testAllCNN(net, t1Test, t1Test_r, gtTest, numTest)

    %% test on the whole dataset with CNN methods(brainweb, 399=239+160)
    test_metric = struct('testTime',0,...
    'AccAll',zeros(numTest, 1),'Acc',zeros(1,2),...
    'JSAll',zeros(numTest, 4),'JS',zeros(4,2),'JS_avg',zeros(1,2),...
    'DSCAll',zeros(numTest, 4),'DSC',zeros(4,2),'DSC_avg',zeros(1,2));

for test_Idx = 1:numTest
    fprintf('%d/%d \t', test_Idx, numTest);
    [test_t1, testInfo] = readimage(t1Test_r, test_Idx);
    test_t1_org = readimage(t1Test, test_Idx);


    startTime = clock; % start eval method time
    test_predlbl = semanticseg(test_t1, net);
    endTime = clock;  % end eval method time
    time = etime(endTime, startTime);

    % resize to original
    test_predlbl = resize2orginal(test_predlbl, test_t1_org);

    % compute metrics
    test_gt = readimage(gtTest, test_Idx);

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
test_metric(1).DSC(:,1) = (mean(test_metric(1).DSCAll,1))';
test_metric(1).DSC(:,2) = (std(test_metric(1).DSCAll,1))';
test_metric(1).DSC_avg(1,1) = mean(test_metric(1).DSC(:,1));
test_metric(1).DSC_avg(1,2) = std(test_metric(1).DSC(:,2));

end