function [imgTrain, imgTest, gtTrain, gtTest] = partitionIBSRData(img,gt)
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(img.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
N = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:N);

% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);

% Create image datastores for training and test.
trainingImages = img.Files(trainingIdx);
testImages = img.Files(testIdx);
imgTrain = imageDatastore(trainingImages);
imgTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = gt.ClassNames;
labelIDs = [1,2,3,4];

% Create pixel label datastores for training and test.
trainingLabels = gt.Files(trainingIdx);
testLabels = gt.Files(testIdx);
gtTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
gtTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end