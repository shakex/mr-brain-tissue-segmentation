function test_metric = testAllCNN_mrbrains(net, t1Test, gtTest, numTest)

test_metric = struct('testTime',0,...
'AccAll',zeros(numTest, 1),'Acc',zeros(1,2),...
'JSAll',zeros(numTest, 4),'JS',zeros(4,2),'JS_avg',zeros(1,2),...
'DSCAll',zeros(numTest, 4),'DSC',zeros(4,2),'DSC_avg',zeros(1,2));

for test_Idx = 1:numTest
    fprintf('%d/%d \t', test_Idx, numTest);
    [test_t1, testInfo] = readimage(t1Test, test_Idx);

    startTime = clock; % start eval method time
    test_predlbl = semanticseg(test_t1, net);
    endTime = clock;  % end eval method time
    time = etime(endTime, startTime);

    % compute metrics
    test_gt = readimage(gtTest, test_Idx);

    testTime = time;
    Acc = sum(sum(double(test_predlbl) == double(test_gt)))./numel(test_predlbl);
    JS = jaccard(double(test_predlbl), double(test_gt));
    DSC = dice(double(test_predlbl), double(test_gt));

    test_metric(1).testTime = test_metric(1).testTime + testTime;
    test_metric(1).AccAll(test_Idx) = Acc;
    test_metric(1).JSAll(test_Idx,:) = JS';
    test_metric(1).DSCAll(test_Idx,:) = DSC';

    fprintf('%.4f \n', sum(sum(double(test_predlbl) == double(test_gt)))./numel(test_predlbl));
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