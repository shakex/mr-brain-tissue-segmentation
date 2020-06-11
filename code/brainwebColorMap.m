function cmap = brainwebColorMap()
% Define the colormap used by brainweb dataset.

% cmap = [
%     0 0 250           % (1)Background
%     125 10 10      % (2)Cerebro-Spinal Fluid
%     239 152 21        % (3)Gray matter
%     250 250 250     % (4)White matter
%     ];

cmap = [
    0 0 0           % (1)Background
    255 255 255      % (2)Cerebro-Spinal Fluid
    92 179 179        % (3)Gray matter
    221 218 93     % (4)White matter
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;

end