function dispSuperpixelGraph(img, L, N, G, scale)

x = zeros(N,1);
y = zeros(N,1);
pixelIdxList = label2idx(L);
for labelVal = 1:N
    x(labelVal) = int16(median(int16(pixelIdxList{labelVal} / size(L,1))));
    y(labelVal) = int16(median(mod(pixelIdxList{labelVal}, size(L,1))));
end

img2 = imresize(img, scale, 'nearest');
boundaryMsk = boundarymask(imresize(L, scale,'nearest'));
figure;
% imshow(imoverlay(img2, boundaryMsk, 'm'), 'InitialMagnification', 100);
imshow(img2, 'InitialMagnification', 100);


% imwrite(imoverlay(img2, boundaryMsk, 'm'), 'C:\Users\shake\Desktop\MRBrainSeg\figs\superpixelrm_pd_c079_slic_bilstm_2000_10_f6.bmp');

% superpixel graph
% idx = label2idx(imresize(L, scale,'nearest'));
% outputImage = zeros(size(img2),'like',img2);
% for labelVal = 1:N
%     labelIdx = idx{labelVal};
%     outputImage(labelIdx) = mean(img2(labelIdx));
% end
% figure; imshow(outputImage,'InitialMagnification',100);


hold on
h = plot(G, 'MarkerSize', 5, 'XData', scale * x, 'YData', scale * y);


h.NodeColor = 'black';
h.LineWidth = 8;
G.Nodes.NodeColors = degree(G);
h.MarkerSize = G.Nodes.NodeColors*5;
colorbar off;


figure;
h = plot(G, 'MarkerSize', 5, 'XData', scale * x, 'YData', scale * y);

h.NodeColor = 'black';
h.LineWidth = 8;
G.Nodes.NodeColors = degree(G);
h.MarkerSize = G.Nodes.NodeColors*5;
colorbar off;

% imwrite(im,'/Users/shake/Documents/master/project/MRBrainSeg/res_ppt/graph.png');

% H = subgraph(G,(1:966));
% figure;
% p = plot(H, 'MarkerSize', 5, 'XData', x(1:966), 'YData', y(1:966));

end