function analyzeGtData(gt, classes)

% plot pixel counts
tbl = countEachLabel(gt);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure; bar(1:numel(classes),frequency);
xticks(1:numel(classes));
xticklabels(tbl.Name);
xtickangle(45);
ylabel('Frequency');

end
