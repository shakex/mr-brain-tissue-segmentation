function pxdsClass = createLabel(pxds, idx, L, N)
% generate label data from label images and superpixel result

labelImage = readimage(pxds,idx);
% labelImage = imresize(double(labelImage),2,'nearest');
pxdsClass = zeros(N,1);

pixelIdxList = label2idx(L);

for labelVal=1:N
    labelRegionIdx = pixelIdxList{labelVal};
    pxdsClass(labelVal) = mode(labelImage(labelRegionIdx));
end
