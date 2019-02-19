function [L, NNew] = removeBackground(img, L, N, alpha)

pixelIdxList = label2idx(L);
threshold = floor(alpha * (max(max(img)) - min(min(img))));
cnt = 1;

for labelVal = 1:N
    if mean(img(pixelIdxList{labelVal})) < threshold
        L(pixelIdxList{labelVal}) = 0;
    else
        L(pixelIdxList{labelVal}) = cnt;
%         NodeGray(cnt,1) = mean(img(pixelIdxList{labelVal}));
        cnt = cnt + 1;
    end
end

NNew = cnt - 1;

end