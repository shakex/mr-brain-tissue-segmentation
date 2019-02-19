function plotClassHistogram(pxds, imds)
% plot histogram for each class
% input: pxds, imds

bgVal = [];
csfVal = [];
gmVal = [];
wmVal = [];
for i=1:numel(imds.Files)
    image = readimage(imds, i);
    label = readimage(pxds, i);
    bgVal = [bgVal;image(label=='BG')];
    csfVal = [csfVal;image(label=='CSF')];
    gmVal = [gmVal;image(label=='GM')];
    wmVal = [wmVal;image(label=='WM')];
end

figure
histogram(bgVal,'BinWidth',1,'Normalization','pdf');
hold on
histogram(csfVal,'BinWidth',1,'Normalization','pdf');
hold on
histogram(gmVal,'BinWidth',1,'Normalization','pdf');
hold on
histogram(wmVal,'BinWidth',1,'Normalization','pdf');

xlabel('Pixel Intensity','FontWeight','bold');
ylabel('Frequency','FontWeight','bold');
legend('BG','CSF','GM','WM','Location','northwest');

end
