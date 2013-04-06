## Run HMC using a Bayesian logit model with a normal Prior N(0, aI) on the 
## Swiss banknote data
using Test
using GeometricMCMC

# Create design matrix X and response variable y from swiss data array
swiss = readdlm("swiss.txt", ' ');
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

# Create setDriftStep function for adjusting the leap step of HMC
leapSteps = [0.4, 1e-3/2, 1e-3, 1e-2, 1e-1, 0.15, 0.2, 0.25, 0.3, 0.35]

setLeapStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, 
  nBurnin::Int, currentStep::Float64) =
  setHmcLeapStep(i, acceptanceRatio, nMcmc, nBurnin, currentStep, leapSteps)

# Create instance malaOpts of HMC options
hmcOpts = HmcOpts(55000, 5000, 10, setLeapStep, eye(nPars));

# Run HMC
mcmc = Array(Float64, nPars, nPars)
z = Array(Float64, nPars, nPars)
linearZvMcmc = Array(Float64, nPars, nPars)
linearCoef = Array(Float64, nPars, nPars)
quadraticZvMcmc = Array(Float64, nPars, nPars)
quadraticCoef = Array(Float64, convert(Int, nPars*(nPars+3)/2), nPars)

try
  mcmc, z = hmc(model, hmcOpts);
  
  # Compute ZV-HMC mean estimators based on linear polynomial
  linearZvMcmc, linearCoef = linearZv(mcmc, z);

  # Compute ZV-HMC mean estimators based on quadratic polynomial
  quadraticZvMcmc, quadraticCoef = quadraticZv(mcmc, z);
catch msg
  println(msg)
end
