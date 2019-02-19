imds = imageDatastore('C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR\t1_');
pxds = imageDatastore('C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR\gt');

for i = 1:numel(imds.Files)
    [img, info] = readimage(imds, i);
    lbl = readimage(pxds, i);
    lbl(lbl==1) = 0;
    msk = logical(lbl);
    out = double(img) .* double(msk);

    saveName = info.Filename(end-9:end);
    imwrite(uint8(out), fullfile('C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR\t1', saveName));
end