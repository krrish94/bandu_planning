% Data provided
data = [
    67, 61;
    68, 63;
    146, 122;
    172, 150
];

% Dataset names
datasets = {'Bandu', 'Custom'};

% Approach name
approach = 'Sym Plan + Geom simpl + SBI-SNRE';

% Ablation study variants
variants = {
    'simplify 10%',
    'simplify 25%',
    'simplify 50%',
    'simplify 90%'
};

% Create the horizontal bar chart
h = barh(data, 'grouped');
set(gca, 'YTickLabel', variants);
xlabel('Values');
ylabel('Ablation Study Variants');
title('Horizontal Bar Chart');

% Add data labels
for i = 1:numel(data)
    text(data(i)+2, i, num2str(data(i)), 'VerticalAlignment', 'middle');
end

% Show grid
grid on;

% Create legend
legend(datasets, 'Location', 'northwest');

% Adjust figure size
figurePosition = [100, 100, 800, 400];
set(gcf, 'Position', figurePosition);