function predImageOrgfuse = createPredOrg(predImage, org, alpha)
% create prediction image with orginal image

[row, col] = size(org);
% figure;imshow(org);
predImageOrg = predImage;
org2 = predImage;
predImageOrgfuse = predImage;
for i=1:row
    for j=1:col
        org2(i,j,1) = org(i,j);
        org2(i,j,2) = org(i,j);
        org2(i,j,3) = org(i,j);
        if (predImageOrg(i,j,1)==0 && predImageOrg(i,j,2)==0 && predImageOrg(i,j,3)==250)
            predImageOrg(i,j,1) = org(i,j);
            predImageOrg(i,j,2) = org(i,j);
            predImageOrg(i,j,3) = org(i,j);
        end
        predImageOrgfuse(i,j,1) = (1-alpha) * org2(i,j,1) + alpha * predImageOrg(i,j,1);
        predImageOrgfuse(i,j,2) = (1-alpha) * org2(i,j,2) + alpha * predImageOrg(i,j,2);
        predImageOrgfuse(i,j,3) = (1-alpha) * org2(i,j,3) + alpha * predImageOrg(i,j,3);
    end
end

