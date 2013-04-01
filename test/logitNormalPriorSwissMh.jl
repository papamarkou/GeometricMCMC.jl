## Run Metropolis-Hastings using a Bayesian logit model with a normal Prior 
## N(0, aI) on the Swiss banknote data

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
data = {"X"=>X, "y"=>y, "priorVar"=>100., "nData"=>nData};

# Create Model instance
model =
  Model(nPars, data, logPrior, logLikelihood, gradLogPosterior, randPrior);

# Create instance mhOpts of Metropolis-Hastings options
mhOpts = MhOpts(55000, 5000, 0.1);

# Run Metropolis-Hastings simulation
mcmc, z = mh(model, mhOpts);

# Compute ZV-MH mean estimators based on linear polynomial
linearZvMcmc, linearCoef = linearZv(mcmc, z);

# Compute ZV-MH mean estimators based on quadratic polynomial
quadraticZvMcmc, quadraticCoef = quadraticZv(mcmc, z);
