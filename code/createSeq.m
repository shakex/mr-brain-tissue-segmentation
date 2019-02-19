function Cell = createSeq(imt1, imt2, impd, G, L, N, featureMode)
% create sequence for training, one cell size: numfeatures x numofneighbors
% parameters:
% numFeatures: number of features, {neighbors, top4 gray}
% I: grayscale image(T1)
% IT2: grayscale image(T2)
% IPD: grayscale image(PD)
% G: superpixel graph
% L: label map of superpixels
% N: actual number of labels

Cell = cell(N,1);
pixelIdxList = label2idx(L);

for labelVal = 1:N
    if featureMode == 1 % node mean gray
        ins = zeros(1, 1);
        labelRegionIdx = pixelIdxList{labelVal};
        ins(1,1) = mean(imt1(labelRegionIdx)); 
        
    elseif featureMode == 2 % multi-modality
        ins = zeros(3, 1);
        labelRegionIdx = pixelIdxList{labelVal};
        ins(1,1) = mean(imt1(labelRegionIdx)); 
        ins(2,1) = mean(imt2(labelRegionIdx)); 
        ins(3,1) = mean(impd(labelRegionIdx));
        
    % good
    elseif featureMode == 3 % adjacent nodes
        seqLen = size(neighbors(G, labelVal),1)+1;
        ins = zeros(1,seqLen);
        labelRegionIdx = pixelIdxList{labelVal};
        ins(1,1) = mean(imt1(labelRegionIdx));
        neighbor = neighbors(G, labelVal);
        for neighborVal = 2:seqLen
            neighborRegionIdx = pixelIdxList{neighbor(neighborVal-1)};
            ins(1, neighborVal) = max(imt1(neighborRegionIdx));
        end
        
    % good
    elseif featureMode == 4 % mod+adj
        seqLen = size(neighbors(G, labelVal),1)+1;
        ins = zeros(3,seqLen);
        labelRegionIdx = pixelIdxList{labelVal};
        ins(1,1) = mean(imt1(labelRegionIdx));
        ins(2,1) = mean(imt2(labelRegionIdx));
        ins(3,1) = mean(impd(labelRegionIdx));
        neighbor = neighbors(G, labelVal);
        for neighborVal = 2:seqLen
            neighborRegionIdx = pixelIdxList{neighbor(neighborVal-1)};
            ins(1, neighborVal) = max(imt1(neighborRegionIdx));
            ins(2, neighborVal) = max(imt2(neighborRegionIdx));
            ins(3, neighborVal) = max(impd(neighborRegionIdx));
        end
    end
     
%       % good
%     elseif featureMode == 5
%         seqLen = size(neighbors(G, labelVal),1)+1;
%         ins = zeros(3,2);
%         labelRegionIdx = pixelIdxList{labelVal};
%         ins(1,1) = mean(imt1(labelRegionIdx));
%         ins(2,1) = mean(imt2(labelRegionIdx));
%         ins(3,1) = mean(impd(labelRegionIdx));
%         neighbor = neighbors(G, labelVal);
%         for neighborVal = 2:seqLen
%             neighborRegionIdx = pixelIdxList{neighbor(neighborVal-1)};
%             ins(1, 2) = ins(1,2) + max(imt1(neighborRegionIdx));
%             ins(2, 2) = ins(2,2) + max(imt2(neighborRegionIdx));
%             ins(3, 2) = ins(3,2) + max(impd(neighborRegionIdx));
%         end
%         
%     elseif featureMode == 8
%         ins = zeros(4, 1);
%         labelRegionIdx = pixelIdxList{labelVal};
%         ins(1,1) = labelVal;
%         ins(2,1) = sum(imt1(labelRegionIdx)); 
%         ins(3,1) = sum(imt2(labelRegionIdx)); 
%         ins(4,1) = sum(impd(labelRegionIdx)); 
%           
%     end
    Cell{labelVal,1} = ins;
end



% for labelVal = 1:N
%     ins = zeros(numFeatures, size(neighbors(G, labelVal),1)+3);
% %     ins = zeros(numFeatures, 3);
%     ins(1,:) = [labelVal labelVal labelVal neighbors(G, labelVal)'];  % feature1
% %     ins(1,:) = [labelVal labelVal labelVal];  % feature1
%     
%     % top4gray: feature2-feature5
%     
%     labelRegionIdx = pixelIdxList{labelVal};  
% %     his = histogram(I(labelRegionIdx));
%     [num, edges] = histcounts(imt1(labelRegionIdx));
%     [numSort, idxSort] = sort(num,'descend');
%     for i = 1:(numFeatures-1)
%         if i <= size(idxSort,2)
%             ins(i+1,1) = (edges(idxSort(i)+1) + edges(idxSort(i))) / 2;
%         else
%             break;
%         end
%     end
%      
%     [num, edges] = histcounts(imt2(labelRegionIdx));
%     [numSort, idxSort] = sort(num,'descend');
%     for i = 1:(numFeatures-1)
%         if i <= size(idxSort,2)
%             ins(i+1,2) = (edges(idxSort(i)+1) + edges(idxSort(i))) / 2;
%         else
%             break;
%         end
%     end
%     
%     [num, edges] = histcounts(impd(labelRegionIdx));
%     [numSort, idxSort] = sort(num,'descend');
%     for i = 1:(numFeatures-1)
%         if i <= size(idxSort,2)
%             ins(i+1,3) = (edges(idxSort(i)+1) + edges(idxSort(i))) / 2;
%         else
%             break;
%         end
%     end
%     
%     neighbor = neighbors(G, labelVal);
%     for neighborVal = 1:size(neighbor,1)
%         labelRegionIdx = pixelIdxList{neighbor(neighborVal)};  
% %         his = histogram(I(labelRegionIdx));
%         [num, edges] = histcounts(imt1(labelRegionIdx));
%         [numSort, idxSort] = sort(num,'descend');
%         for i = 1:(numFeatures-1)
%             if i <= size(idxSort,2)
%                 ins(i+1,neighborVal+3) = (edges(idxSort(i)+1) + edges(idxSort(i))) / 2;
%             else
%                 break;
%             end
%         end
%     end
%     Cell{labelVal,1} = ins;
end
