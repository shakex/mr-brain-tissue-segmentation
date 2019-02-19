% plot superpixel with nosie
slicNum = 2500;
compactness = 10;
for trainIdx = 2:2
    t1 = readimage(t1Train, trainIdx);
    t1ir = readimage(t1irTrain, trainIdx);
    t2flair = readimage(t2flairTrain, trainIdx);
    
    % add noise
    sp5 = imnoise(t1,'salt & pepper',0.05);
    sp10 = imnoise(t1,'salt & pepper',0.10);
    sp15 = imnoise(t1,'salt & pepper',0.15);
    sp20 = imnoise(t1,'salt & pepper',0.20);
    figure;
    subplot(2,3,1); imshow(t1); title('salt&pepper 0%');
    subplot(2,3,2); imshow(sp5); title('salt&pepper 5%');
    subplot(2,3,3); imshow(sp10); title('salt&pepper 10%');
    subplot(2,3,4); imshow(sp15); title('salt&pepper 15%');
    subplot(2,3,5); imshow(sp20); title('salt&pepper 20%');
    
    % slic
    [L, N] = superpixels(t1, slicNum, 'Compactness', compactness);
    [L_sp5, N_sp5] = superpixels(sp5, slicNum, 'Compactness', compactness);
    [L_sp10, N_sp10] = superpixels(sp10, slicNum, 'Compactness', compactness);
    [L_sp15, N_sp15] = superpixels(sp15, slicNum, 'Compactness', compactness);
    [L_sp20, N_sp20] = superpixels(sp20, slicNum, 'Compactness', compactness);
    
    figure;
    M = boundarymask(imresize(L, 4,'nearest'));
    t1_4 = imresize(t1, 4,'nearest');
    subplot(2,3,1); imshow(imoverlay(t1_4, M, 'y'));title('salt&pepper 0%');
    M_sp5 = boundarymask(imresize(L_sp5, 4,'nearest'));
    sp5_4 = imresize(sp5, 4,'nearest');
    subplot(2,3,2); imshow(imoverlay(sp5_4, M_sp5, 'y'));title('salt&pepper 5%');
    M_sp10 = boundarymask(imresize(L_sp10, 4,'nearest'));
    sp10_4 = imresize(sp10, 4,'nearest');
    subplot(2,3,3); imshow(imoverlay(sp10_4, M_sp10, 'y'));title('salt&pepper 10%');
    M_sp15 = boundarymask(imresize(L_sp15, 4,'nearest'));
    sp15_4 = imresize(sp15, 4,'nearest');
    subplot(2,3,4); imshow(imoverlay(sp15_4, M_sp15, 'y'));title('salt&pepper 15%');
    M_sp20 = boundarymask(imresize(L_sp20, 4,'nearest'));
    sp20_4 = imresize(sp20, 4,'nearest');
    subplot(2,3,5); imshow(imoverlay(sp20_4, M_sp20, 'y'));title('salt&pepper 20%');
    
    % remove background
    [L, N] = removeBackground(t1, L, N, 0.05);
    [L_sp5, N_sp5] = removeBackground(sp5, L_sp5, N_sp5, 0.05);
    [L_sp10, N_sp10] = removeBackground(sp10, L_sp10, N_sp10, 0.05);
    [L_sp15, N_sp15] = removeBackground(sp15, L_sp15, N_sp15, 0.05);
    [L_sp20, N_sp20] = removeBackground(sp20, L_sp20, N_sp20, 0.05);
    
    figure;
    M = boundarymask(imresize(L, 4,'nearest'));
    t1_4 = imresize(t1, 4,'nearest');
    subplot(2,3,1); imshow(imoverlay(t1_4, M, 'y'));title('salt&pepper 0%');
    M_sp5 = boundarymask(imresize(L_sp5, 4,'nearest'));
    sp5_4 = imresize(sp5, 4,'nearest');
    subplot(2,3,2); imshow(imoverlay(sp5_4, M_sp5, 'y'));title('salt&pepper 5%');
    M_sp10 = boundarymask(imresize(L_sp10, 4,'nearest'));
    sp10_4 = imresize(sp10, 4,'nearest');
    subplot(2,3,3); imshow(imoverlay(sp10_4, M_sp10, 'y'));title('salt&pepper 10%');
    M_sp15 = boundarymask(imresize(L_sp15, 4,'nearest'));
    sp15_4 = imresize(sp15, 4,'nearest');
    subplot(2,3,4); imshow(imoverlay(sp15_4, M_sp15, 'y'));title('salt&pepper 15%');
    M_sp20 = boundarymask(imresize(L_sp20, 4,'nearest'));
    sp20_4 = imresize(sp20, 4,'nearest');
    subplot(2,3,5); imshow(imoverlay(sp20_4, M_sp20, 'y'));title('salt&pepper 20%');
    
%% gaussion
    g1 = imnoise(t1,'gaussian',0,0.01);
    g3 = imnoise(t1,'gaussian',0,0.03);
    g5 = imnoise(t1,'gaussian',0,0.05);
    g7 = imnoise(t1,'gaussian',0,0.07);
    g9 = imnoise(t1,'gaussian',0,0.09);
    
    figure;
    subplot(2,3,1); imshow(t1); title('Gaussian 0%');
    subplot(2,3,2); imshow(g1); title('Gaussian 1%');
    subplot(2,3,3); imshow(g3); title('Gaussian 3%');
    subplot(2,3,4); imshow(g5); title('Gaussian 5%');
    subplot(2,3,5); imshow(g7); title('Gaussian 7%');
    subplot(2,3,6); imshow(g9); title('Gaussian 9%');
    
    % slic
    [L, N] = superpixels(t1, slicNum, 'Compactness', compactness);
    [L_g1, N_g1] = superpixels(g1, slicNum, 'Compactness', compactness);
    [L_g3, N_g3] = superpixels(g3, slicNum, 'Compactness', compactness);
    [L_g5, N_g5] = superpixels(g5, slicNum, 'Compactness', compactness);
    [L_g7, N_g7] = superpixels(g7, slicNum, 'Compactness', compactness);
    [L_g9, N_g9] = superpixels(g9, slicNum, 'Compactness', compactness);
    
    figure;
    M = boundarymask(imresize(L, 4,'nearest'));
    t1_4 = imresize(t1, 4,'nearest');
    subplot(2,3,1); imshow(imoverlay(t1_4, M, 'y'));title('Gaussion 0%');
    M_g1 = boundarymask(imresize(L_g1, 4,'nearest'));
    g1_4 = imresize(g1, 4,'nearest');
    subplot(2,3,2); imshow(imoverlay(g1_4, M_g1, 'y'));title('Gaussion 1%');
    M_g3 = boundarymask(imresize(L_g3, 4,'nearest'));
    g3_4 = imresize(g3, 4,'nearest');
    subplot(2,3,3); imshow(imoverlay(g3_4, M_g3, 'y'));title('Gaussion 3%');
    M_g5 = boundarymask(imresize(L_g5, 4,'nearest'));
    g5_4 = imresize(g5, 4,'nearest');
    subplot(2,3,4); imshow(imoverlay(g5_4, M_g5, 'y'));title('Gaussion 5%');
    M_g7 = boundarymask(imresize(L_g7, 4,'nearest'));
    g7_4 = imresize(g7, 4,'nearest');
    subplot(2,3,5); imshow(imoverlay(g7_4, M_g7, 'y'));title('Gaussion 7%');
    M_g9 = boundarymask(imresize(L_g9, 4,'nearest'));
    g9_4 = imresize(g9, 4,'nearest');
    subplot(2,3,5); imshow(imoverlay(g9_4, M_g9, 'y'));title('Gaussion 9%');
    
    % remove background
    [L, N] = removeBackground(t1, L, N, 0.05);
    [L_g1, N_g1] = removeBackground(g1, L_g1, N_g1, 0.05);
    [L_g3, N_g3] = removeBackground(g3, L_g3, N_g3, 0.05);
    [L_g5, N_g5] = removeBackground(g5, L_g5, N_g5, 0.05);
    [L_g7, N_g7] = removeBackground(g7, L_g7, N_g7, 0.05);
    [L_g9, N_g9] = removeBackground(g9, L_g9, N_g9, 0.05);
    
    figure;
    M = boundarymask(imresize(L, 4,'nearest'));
    t1_4 = imresize(t1, 4,'nearest');
    subplot(2,3,1); imshow(imoverlay(t1_4, M, 'y'));title('Gaussion 0%');
    M_g1 = boundarymask(imresize(L_g1, 4,'nearest'));
    g1_4 = imresize(g1, 4,'nearest');
    subplot(2,3,2); imshow(imoverlay(g1_4, M_g1, 'y'));title('Gaussion 1%');
    M_g3 = boundarymask(imresize(L_g3, 4,'nearest'));
    g3_4 = imresize(g3, 4,'nearest');
    subplot(2,3,3); imshow(imoverlay(g3_4, M_g3, 'y'));title('Gaussion 3%');
    M_g5 = boundarymask(imresize(L_g5, 4,'nearest'));
    g5_4 = imresize(g5, 4,'nearest');
    subplot(2,3,4); imshow(imoverlay(g5_4, M_g5, 'y'));title('Gaussion 5%');
    M_g7 = boundarymask(imresize(L_g7, 4,'nearest'));
    g7_4 = imresize(g7, 4,'nearest');
    subplot(2,3,5); imshow(imoverlay(g7_4, M_g7, 'y'));title('Gaussion 7%');
    M_g9 = boundarymask(imresize(L_g9, 4,'nearest'));
    g9_4 = imresize(g9, 4,'nearest');
    subplot(2,3,5); imshow(imoverlay(g9_4, M_g9, 'y'));title('Gaussion 9%');
    

end
