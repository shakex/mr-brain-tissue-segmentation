function Cell = createPixelSeq(imt1, imt2, impd, featureMode)

N = numel(imt1);
[row, col] = size(imt1);
Cell = cell(N,1);
for i = 1:row
    for j = 1:col
        if featureMode == 1
            % pixel gray(featureNum=1)
            ins = zeros(1,1);
            ins(1,1) = imt1(i, j);
         
        elseif featureMode == 2
            % add modality(featurenNum=3)
            ins = zeros(3,1);
            ins(1,1) = imt1(i, j); 
            ins(2,1) = imt2(i, j);
            ins(3,1) = impd(i, j);
                  
        elseif featureMode == 3
            % add neighbors(featureNum=1)
            if (i==1) || (j==1) || (i==row) || (j==col)
                ins = zeros(1, 1);
                ins(1,1) = imt1(i, j);
            else
                ins = zeros(1, 9);
                ins(1,1) = imt1(i,j);
                ins(1,2) = imt1(i-1,j-1);
                ins(1,3) = imt1(i-1,j);
                ins(1,4) = imt1(i-1,j+1);
                ins(1,5) = imt1(i,j+1);
                ins(1,6) = imt1(i+1,j+1);
                ins(1,7) = imt1(i+1,j);
                ins(1,8) = imt1(i+1,j-1);
                ins(1,9) = imt1(i,j-1);
            end
        
            
        
        elseif featureMode == 4
            % add modality&neighbors(featureNum=3)
            if (i==1) || (j==1) || (i==row) || (j==col)
                ins = zeros(3, 1);
                ins(1,1) = imt1(i,j);
                ins(2,1) = imt2(i,j);
                ins(3,1) = impd(i,j);
            else
                ins = zeros(3, 9);
                ins(1,1) = imt1(i,j);
                ins(1,2) = imt1(i-1,j-1);
                ins(1,3) = imt1(i-1,j);
                ins(1,4) = imt1(i-1,j+1);
                ins(1,5) = imt1(i,j+1);
                ins(1,6) = imt1(i+1,j+1);
                ins(1,7) = imt1(i+1,j);
                ins(1,8) = imt1(i+1,j-1);
                ins(1,9) = imt1(i,j-1);
                ins(2,1) = imt2(i,j);
                ins(2,2) = imt2(i-1,j-1);
                ins(2,3) = imt2(i-1,j);
                ins(2,4) = imt2(i-1,j+1);
                ins(2,5) = imt2(i,j+1);
                ins(2,6) = imt2(i+1,j+1);
                ins(2,7) = imt2(i+1,j);
                ins(2,8) = imt2(i+1,j-1);
                ins(2,9) = impd(i,j-1);
                ins(3,1) = impd(i,j);
                ins(3,2) = impd(i-1,j-1);
                ins(3,3) = impd(i-1,j);
                ins(3,4) = impd(i-1,j+1);
                ins(3,5) = impd(i,j+1);
                ins(3,6) = impd(i+1,j+1);
                ins(3,7) = impd(i+1,j);
                ins(3,8) = impd(i+1,j-1);
                ins(3,9) = impd(i,j-1);
            end
            
        end
        
        Cell{(i-1)*col+j,1} = ins;
        
    end
end

end