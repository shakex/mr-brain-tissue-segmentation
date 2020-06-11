% imds = imageDatastore('C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR\t1_');
% pxds = imageDatastore('C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR\gt');
% 
% for i = 1:numel(imds.Files)
%     [img, info] = readimage(imds, i);
%     lbl = readimage(pxds, i);
%     lbl(lbl==1) = 0;
%     msk = logical(lbl);
%     out = double(img) .* double(msk);
% 
%     saveName = info.Filename(end-9:end);
%     imwrite(uint8(out), fullfile('C:\Users\shake\Desktop\MRBrainSeg\dataset\IBSR\t1', saveName));
% end


imds = imageDatastore('C:\Users\shake\Desktop\Seg_0312\brainweb\gt');

for i = 1:numel(imds.Files)
    [img, info] = readimage(imds, i);
    [row, col] = size(img);
    for m = 1:row
        for n = 1:col
            if img(m,n)==1
                img(m,n)=0;
            elseif img(m,n)==2
                img(m,n)=1;
            elseif img(m,n)==3
                img(m,n)=2;
            else 
                img(m,n)=3;
            end
        end
    end
    name = info.Filename(end-7:end);
    imwrite(img, fullfile('C:\Users\shake\Desktop\Seg_0312\brainweb\gt_',name));
    
end
