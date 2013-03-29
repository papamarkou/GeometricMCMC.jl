# Create design matrix X and response variable y from swiss data array
# swiss = readdlm("../test/swiss.txt", ' ');
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
