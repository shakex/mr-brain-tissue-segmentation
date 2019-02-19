function pixelTrain = extractFeaturesTrain(t1Train, t2Train, pdTrain, gtTrain)

Array = [];
for trainIdx = 1
    fprintf('extracting features: %d/1\n', trainIdx);
    imt1 = readimage(t1Train, trainIdx);
    imt2 = readimage(t2Train, trainIdx);
    impd = readimage(pdTrain, trainIdx);
    
%     imt1 = imnoise(imt1,'gaussian',0,var);
%     imt2 = imnoise(imt2,'gaussian',0,var);
%     impd = imnoise(impd,'gaussian',0,var);
    
    labelImage = readimage(gtTrain, trainIdx);

    [row, col] = size(imt1);
    ins = zeros(numel(imt1), 4);
    for i=1:row
        for j=1:col
            ins((i - 1) * col + j, 1) = imt1(i, j); 
            ins((i - 1) * col + j, 2) = imt2(i, j); 
            ins((i - 1) * col + j, 3) = impd(i, j);
            ins((i - 1) * col + j, 4) = labelImage(i, j);
        end
    end
    Array = [Array;ins];
end

pixelTrain = array2table(Array);

end
