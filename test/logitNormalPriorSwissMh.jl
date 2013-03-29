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
data = {"X"=>X, "y"=>y, "priorVar"=>100.};

# Create Model instance
model =
  Model(nPars, data, logPrior, logLikelihood, gradLogPosterior, randPrior);

# Create instance mhOpts of Metropolis-Hastings options
mhOpts = MhOpts(55000, 5000, 0.1);

# Run Metropolis-Hastings simulation
mhOut = mh(model, mhOpts);

# Compute ZV-RMHMC estimates based on linear polynomial
[BZvL, polCoefL] = linearZv(B, Z);

# Compute ZV-RMHMC estimates based on quadratic polynomial
[BZvQ, polCoefQ] = quadraticZv(B, Z);

# Save output in file
save(['./examples/logitNormalPriorSwiss/output/' ...
  'logitNormalPriorSwissMetropolis.' ...
  'nMcmc' num2str(metropolisParameters(1)) '.' ...
  'nBurnIn' num2str(metropolisParameters(2)) '.' ...
  'widthCorrection' num2str(metropolisParameters(3)) '.mat'], ...
  'B', 'BZvL', 'BZvQ')
