function Cell = createSeq(im,lab,hsv,PPG,G,L,N)

Cell = cell(N,1);
pixelIdxList = label2idx(L);

for labelVal = 1:N
    seqLen = size(neighbors(G, labelVal),1)+1;
    ins = zeros(7,seqLen);
    labelRegionIdx = pixelIdxList{labelVal};
    l = lab(:,:,1);
    a = lab(:,:,2);
    b = lab(:,:,3);
    h = hsv(:,:,1);
    s = hsv(:,:,2);
    v = hsv(:,:,3);
    
    ins(1,1) = mean(l(labelRegionIdx));
    ins(2,1) = mean(a(labelRegionIdx));
    ins(3,1) = mean(b(labelRegionIdx));
    ins(4,1) = mean(h(labelRegionIdx));
    ins(5,1) = mean(s(labelRegionIdx));
    ins(6,1) = mean(v(labelRegionIdx));
    ins(7,1) = mean(PPG(labelRegionIdx));
    neighbor = neighbors(G, labelVal);
    for neighborVal = 2:seqLen
        neighborRegionIdx = pixelIdxList{neighbor(neighborVal-1)};
        ins(1, neighborVal) = max(l(neighborRegionIdx));
        ins(2, neighborVal) = max(a(neighborRegionIdx));
        ins(3, neighborVal) = max(b(neighborRegionIdx));
        ins(4, neighborVal) = max(h(neighborRegionIdx));
        ins(5, neighborVal) = max(s(neighborRegionIdx));
        ins(6, neighborVal) = max(v(neighborRegionIdx));
        ins(7, neighborVal) = max(PPG(neighborRegionIdx));
    end
          
    Cell{labelVal,1} = ins;
end

end
