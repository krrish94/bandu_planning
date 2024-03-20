% Data provided
data = [
    197, 198, 0, 0, 0, 0;
    178, 182, 0, 0, 0, 0;
    11, 68, 121, 51, 174, 172;
    9, 61, 102, 48, 147, 150
];

% Dataset names
datasets = {'Blocks', 'Cuboids', 'Bandu', 'Custom'};

% Approach names
approaches = {
    'Randomized symbolic action sampler',
    'Sym plan',
    'Sym plan + Geom simpl',
    'Sym plan + Geom simpl + SBI-SNPE',
    'Sym plan + Geom simpl + SBI-SNLE',
    'Sym plan + Geom simpl + SBI-SNRE'
};

% Create the bar plot with dataset grouping
h = bar(data, 'grouped');
set(gca, 'XTickLabel', datasets, 'XTickLabelRotation', 45);
xlabel('Datasets');
ylabel('Values');
title('Bar Plot with Legend (Grouped by Dataset)');

% Add data labels
for i = 1:numel(data(:))
    if data(i) > 0
        text(i, data(i) + 5, num2str(data(i)), 'HorizontalAlignment', 'center');
    end
end

% Show grid
grid on;

% Create legend
legend(h, approaches);

% Adjust figure size
figurePosition = [100, 100, 1000, 500];
set(gcf, 'Position', figurePosition);