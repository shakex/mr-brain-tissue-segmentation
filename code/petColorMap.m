function cmap = petColorMap()
% Define the colormap used by brainweb dataset.

cmap = [
    255 255 0      % (1)FG
    0 0 0          % (2)BG
    255 0 0        % (3)UC
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;

end