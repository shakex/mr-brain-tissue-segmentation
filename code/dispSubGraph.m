function dispSubGraph(img, L, N, G, snode, enode, scale)

x = zeros(N,1);
y = zeros(N,1);
pixelIdxList = label2idx(L);
for labelVal = 1:N
    x(labelVal) = int16(median(int16(pixelIdxList{labelVal} / size(L,1))));
    y(labelVal) = int16(median(mod(pixelIdxList{labelVal}, size(L,1))));
end

img2 = imresize(img, scale, 'nearest');
boundaryMsk = boundarymask(imresize(L, scale,'nearest'));

H = subgraph(G,(snode:enode));

figure;
imshow(imoverlay(img2, boundaryMsk, 'y'), 'InitialMagnification', 100);
hold on
p = plot(H, 'MarkerSize', 5, 'XData', scale * x(snode:enode), 'YData', scale * y(snode:enode));

end