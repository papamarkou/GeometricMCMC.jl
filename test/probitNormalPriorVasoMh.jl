## Run Metropolis-Hastings using a Bayesian probit model with a Normal prior 
## N(0, priorVar*I) on the vasoconstriction data

using Distributions, GeometricMCMC

include("vaso.jl")
include("probitNormalPrior.jl")

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
