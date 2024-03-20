% Data provided
data = [
    172, 150;
    168, 147;
    160, 141;
    133, 127
];

% Dataset names
datasets = {'Bandu', 'Custom'};

% Approach name
approach = 'Sym Plan + Geom simpl + SBI-SNRE';

% Variants of the approach
variants = {
    'w/ GT physical param',
    'w/ phys param 10% var',
    'w/ phys param 25% var',
    'w/ phys param 50% var'
};

% Create the horizontal bar chart
h = barh(data, 'grouped');
set(gca, 'YTickLabel', variants);
xlabel('Values');
ylabel('Variants of the Approach');
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