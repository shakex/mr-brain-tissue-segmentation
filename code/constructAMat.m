function adjMat = constructAMat(L, N)
% create the adjacency matrix of superpixels
% parameters:
% L: label map of superpixels
% N: actual number of labels

[row, col] = size(L);
adjMat = zeros(N);
for i = 2:row-1
    for j = 2:col-1
        if L(i,j) ~= 0 && L(i-1,j) ~= 0 && L(i+1,j) ~= 0 && L(i,j-1) ~= 0 && L(i,j+1) ~= 0
            if L(i,j) ~= L(i-1,j)
                adjMat(L(i,j), L(i-1,j)) = 1;
                adjMat(L(i-1,j), L(i,j)) = 1;
            end
            if L(i,j) ~= L(i+1,j)
                adjMat(L(i,j), L(i+1,j)) = 1;
                adjMat(L(i+1,j), L(i,j)) = 1;
            end
            if L(i,j) ~= L(i,j-1)
                adjMat(L(i,j-1), L(i,j)) = 1;
                adjMat(L(i,j), L(i,j-1)) = 1;
            end
            if L(i,j) ~= L(i,j+1)
                adjMat(L(i,j), L(i,j+1)) = 1;
                adjMat(L(i,j+1), L(i,j)) = 1;
            end
        end  
    end
end
