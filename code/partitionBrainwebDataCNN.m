function [t1Train_r, t1Test_r, t1Train_r_rgb, t1Test_r_rgb, gtTrain_r, gtTest_r] = partitionBrainwebDataCNN(t1_r,t1_r_rgb,gt_r)
% Partition BrainWeb data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(t1_r.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
N = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:N);

% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);

% Create image datastores for training and test.
trainingImagest1r = t1_r.Files(trainingIdx);
trainingImagest1rgb = t1_r_rgb.Files(trainingIdx);
testImagest1r = t1_r.Files(testIdx);
testImagest1rgb = t1_r_rgb.Files(testIdx);

t1Train_r = imageDatastore(trainingImagest1r);
t1Train_r_rgb = imageDatastore(trainingImagest1rgb);
t1Test_r = imageDatastore(testImagest1r);
t1Test_r_rgb = imageDatastore(testImagest1rgb);

% Extract class and label IDs info.
classes = gt_r.ClassNames;
labelIDs = [1,2,3,4];

% Create pixel label datastores for training and test.
trainingLabels = gt_r.Files(trainingIdx);
testLabels = gt_r.Files(testIdx);
gtTrain_r = pixelLabelDatastore(trainingLabels, classes, labelIDs);
gtTest_r = pixelLabelDatastore(testLabels, classes, labelIDs);
end