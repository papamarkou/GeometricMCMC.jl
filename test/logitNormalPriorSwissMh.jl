# Create design matrix X and response variable y from swiss data array
# swiss = readdlm("GeometricMCMC/examples/data/swiss.txt", ' ');
swiss = readdlm("../examples/data/swiss.txt", ' ');
covariates = swiss[:, 1:end-1];
nData, nPars = size(covariates);

covariates = (bsxfun(-, covariates, mean(covariates, 1))
  ./repmat(std(covariates, 1), nData, 1));

polynomialOrder = 1;
X = zeros(nData, nPars*polynomialOrder);
for i = 1:polynomialOrder
  X[:, ((i-1)*nPars+1):i*nPars] = covariates.^i;
end

y = swiss[:, end];

# Create data dictionary
data = {"X"=>X, "y"=>y, "priorVar"=>100., "nPars"=>nPars};

# Create Model instance
model =
  Model(nPars, data, logPrior, logLikelihood, gradLogPosterior, randPrior);

options.initialParameters = zeros(1, model.np);

options.nMcmc = 55000;
options.nBurnIn = 5000;

options.proposalWidthCorrection = 0.1;
options.monitorRate = 100;

metropolisOutput = metropolis(model, options);
B = metropolisOutput{1};
Z = metropolisOutput{2};
metropolisParameters = metropolisOutput{3};

%% Compute ZV-RMHMC estimates based on linear polynomial
[BZvL, polCoefL] = linearZv(B, Z);

%% Compute ZV-RMHMC estimates based on quadratic polynomial
[BZvQ, polCoefQ] = quadraticZv(B, Z);

%% Save numerical output
save(['./examples/logitNormalPriorSwiss/output/' ...
  'logitNormalPriorSwissMetropolis.' ...
  'nMcmc' num2str(metropolisParameters(1)) '.' ...
  'nBurnIn' num2str(metropolisParameters(2)) '.' ...
  'widthCorrection' num2str(metropolisParameters(3)) '.mat'], ...
  'B', 'BZvL', 'BZvQ')

%% Generate and save the plots of wZvL paths
fig = figure('PaperUnits','centimeters');
set(fig, 'PaperSize', [18 4] );
set(fig, 'PaperPositionMode', 'manual');
set(fig, 'PaperPosition', [0 0 18 4]);
set(fig, 'renderer', 'painters');
hold on

for i = 1:d
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
  'logitNormalPriorSwissMetropolis.' ...
  'nMcmc' num2str(metropolisParameters(1)) '.' ...
  'nBurnIn' num2str(metropolisParameters(2)) '.' ...
  'widthCorrection' num2str(metropolisParameters(3)) ...
  '.theta' num2str(i) '.linearP.eps']);
end

%% Generate and save the plots of wZvQ paths
fig = figure('PaperUnits','centimeters');
set(fig, 'PaperSize', [18 4] );
set(fig, 'PaperPositionMode', 'manual');
set(fig, 'PaperPosition', [0 0 18 4]);
set(fig, 'renderer', 'painters');
hold on

for i = 1:d
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
  'logitNormalPriorSwissMetropolis.' ...
  'nMcmc' num2str(metropolisParameters(1)) '.' ...
  'nBurnIn' num2str(metropolisParameters(2)) '.' ...
  'widthCorrection' num2str(metropolisParameters(3)) ...
  '.theta' num2str(i) '.quadraticP.eps']);
end
