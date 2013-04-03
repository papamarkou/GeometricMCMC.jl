%% Draw random numbers
rng('shuffle', 'twister');

%% Load and prepare data
load('./examples/data/swiss.mat');
y = data(:,end);
data(:, end) = [];
[n, d] = size(data);
        
%% Standardise data
data = (data-repmat(mean(data), n, 1))./repmat(std(data), n, 1);
        
%% Create polynomial basis
polynomialOrder = 1;
X = zeros(n, d*polynomialOrder);
for i = 1:polynomialOrder
  X(:, ((i-1)*d+1):i*d) = data.^i;
end

%% Run logitNormalPriorSwissHmc
model = LogitNormalPrior(X, y, 100);

options.nMcmc = 55000;
options.nBurnIn = 5000;

options.massMatrix = diag(ones(model.np, 1));
options.nLeap = 10;

options.leapStep = 0.4;
options.leapStep01 = 1e-3/2;
options.leapStep02 = 1e-3;
options.leapStep03 = 1e-2;
options.leapStep04 = 1e-1;
options.leapStep05 = 0.15;
options.leapStep06 = 0.2;
options.leapStep07 = 0.25;
options.leapStep08 = 0.3;
options.leapStep09 = 0.35;

options.monitorRate = 100;

hmcOutput = hmc(model, options);
B = hmcOutput{1};
Z = hmcOutput{2};
hmcParameters = hmcOutput{3};

%% Compute ZV-HMC estimates based on linear polynomial
[BZvL, polCoefL] = linearZv(B, Z);

%% Compute ZV-HMC estimates based on quadratic polynomial
[BZvQ, polCoefQ] = quadraticZv(B, Z);

%% Save numerical output
save(['./examples/logitNormalPriorSwiss/output/' ...
  'logitNormalPriorSwissHmc.' ...
  'nMcmc' num2str(hmcParameters(1)) '.' ...
  'nBurnIn' num2str(hmcParameters(2)) '.' ...
  'nLeap' num2str(hmcParameters(3)) '.' ...
  'leapStep' num2str(hmcParameters(4)) '.mat'], ...
  'B', 'BZvL', 'BZvQ')

%% Generate and save the plots of wZvL paths
fig = figure('PaperUnits','centimeters');
set(fig, 'PaperSize', [18 4] );
set(fig, 'PaperPositionMode', 'manual');
set(fig, 'PaperPosition', [0 0 18 4]);
set(fig, 'renderer', 'painters');
hold on

for i = 1:model.np
  plot01 = plot(B(:,i), 'color', 'blue', 'LineWidth', 0.2);
  title01 = title('1^{st} degree P');
  ylabel01 = ylabel(['\theta_' num2str(i)]);
  hold on
  plot(BZvL(:,i), 'color', 'red', 'LineWidth', 0.2);
  
  set([title01 ylabel01], 'FontName', 'Times-Roman', 'FontSize', 9);
  set(gca, ...
  'FontName', 'Times-Roman', ...
  'FontSize', 9, ...
  'Box', 'on', ...
  'TickDir', 'in', ...
  'TickLength', [.02 .02], ...
  'XMinorTick', 'on', ...
  'YMinorTick', 'on', ...
  'LineWidth', 0.2);

  hold off

  print(fig, '-depsc', ...
  ['./examples/logitNormalPriorSwiss/output/' ...
  'logitNormalPriorSwissHmc.' ...
  'nMcmc' num2str(hmcParameters(1)) '.' ...
  'nBurnIn' num2str(hmcParameters(2)) '.' ...
  'nLeap' num2str(hmcParameters(3)) '.' ...
  'leapStep' num2str(hmcParameters(4)) '.' ...
  'theta' num2str(i) '.linearP.eps']);
end

%% Generate and save the plots of wZvQ paths
fig = figure('PaperUnits','centimeters');
set(fig, 'PaperSize', [18 4] );
set(fig, 'PaperPositionMode', 'manual');
set(fig, 'PaperPosition', [0 0 18 4]);
set(fig, 'renderer', 'painters');
hold on

for i = 1:model.np
  plot01 = plot(B(:,i), 'color', 'blue', 'LineWidth', 0.2);
  title01 = title('2^{nd} degree P');
  ylabel01 = ylabel(['\theta_' num2str(i)]);
  hold on
  plot(BZvQ(:,i), 'color', 'red', 'LineWidth', 0.2);
  
  set([title01 ylabel01], 'FontName', 'Times-Roman', 'FontSize', 9);
  set(gca, ...
  'FontName', 'Times-Roman', ...
  'FontSize', 9, ...
  'Box', 'on', ...
  'TickDir', 'in', ...
  'TickLength', [.02 .02], ...
  'XMinorTick', 'on', ...
  'YMinorTick', 'on', ...
  'LineWidth', 0.2);

  hold off

  print(fig, '-depsc', ...
  ['./examples/logitNormalPriorSwiss/output/' ...
  'logitNormalPriorSwissHmc.' ...
  'nMcmc' num2str(hmcParameters(1)) '.' ...
  'nBurnIn' num2str(hmcParameters(2)) '.' ...
  'nLeap' num2str(hmcParameters(3)) '.' ...
  'leapStep' num2str(hmcParameters(4)) '.' ...
  'theta' num2str(i) '.quadraticP.eps']);
end
