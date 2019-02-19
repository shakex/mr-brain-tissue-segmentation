% IBSR experiments

%% dataset setting
brainwebFolder = 'C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR';
classes = [
    "BG"
    "CSF"
    "GM"
    "WM"
    ];
cmap = brainwebColorMap;
gtIDs = [1 2 3 4];

t1Dir = fullfile(brainwebFolder, 't1');
gtDir = fullfile(brainwebFolder, 'gt');

t1 = imageDatastore(t1Dir);
gt = pixelLabelDatastore(gtDir, classes, gtIDs);

[t1Train, t1Test, gtTrain, gtTest] = partitionIBSRData(t1, gt);
numTrain = numel(t1Train.Files);
numTest = numel(t1Test.Files);

% display sample images
sampleIm = readimage(t1, 2700);
samplegt = readimage(gt, 2700);
sampleOverlay = labeloverlay(sampleIm, samplegt, 'ColorMap', cmap, 'Transparency',0.6);
imshow(sampleOverlay);
pixelLabelColorbar(cmap, classes);

% analyze dataset statistics
% analyzeGtData(gt, classes);
% plotClassHistogram(gtTrain, pdTrain);

%%
