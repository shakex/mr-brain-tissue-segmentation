function [t1Train, t1Test, t2Train, t2Test, pdTrain, pdTest, gtTrain, gtTest] = partitionBrainWebData(t1,t2,pd,gt)
% Partition BrainWeb data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(t1.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
N = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:N);

% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);

% Create image datastores for training and test.
trainingImages = t1.Files(trainingIdx);
trainingImagesT2 = t2.Files(trainingIdx);
trainingImagesPD = pd.Files(trainingIdx);
testImages = t1.Files(testIdx);
testImagesT2 = t2.Files(testIdx);
testImagesPD = pd.Files(testIdx);
t1Train = imageDatastore(trainingImages);
t2Train = imageDatastore(trainingImagesT2);
pdTrain = imageDatastore(trainingImagesPD);
t1Test = imageDatastore(testImages);
t2Test = imageDatastore(testImagesT2);
pdTest = imageDatastore(testImagesPD);

% Extract class and label IDs info.
classes = gt.ClassNames;
labelIDs = [1,2,3,4];

% Create pixel label datastores for training and test.
trainingLabels = gt.Files(trainingIdx);
testLabels = gt.Files(testIdx);
gtTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
gtTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end