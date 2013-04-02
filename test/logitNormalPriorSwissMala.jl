## Run MALA using a Bayesian logit model with a normal Prior N(0, aI) on the 
## Swiss banknote data

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

# Create setDriftStep function for adjusting the drift step of MALA
driftSteps = [1, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25]

setDriftStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, 
  nBurnin::Int, currentStep::Float64) =
  setMalaDriftStep(i, acceptanceRatio, nMcmc, nBurnin, currentStep, driftSteps)

# Create instance malaOpts of MALA options
malaOpts = MalaOpts(55000, 5000, setDriftStep);

try
  # Run MALA
  mcmc, z = mala(model, malaOpts);
  
  # Compute ZV-MALA mean estimators based on linear polynomial
  linearZvMcmc, linearCoef = linearZv(mcmc, z);

  # Compute ZV-MALA mean estimators based on quadratic polynomial
  quadraticZvMcmc, quadraticCoef = quadraticZv(mcmc, z);
catch msg
  println(msg)
end
