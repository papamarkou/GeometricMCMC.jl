## Run SMMALA using a Bayesian probit model with a Normal prior N(0, priorVar*I)
## on the Swiss banknote data

using Distributions, GeometricMCMC

include("vaso.jl")
include("probitNormalPrior.jl")
  
# Create Model instance
model = Model(nPars, data, logPrior, logLikelihood, gradLogPosterior,
  tensor, randPrior);

# Create setDriftStep function for adjusting the drift step of SMMALA
driftSteps = [1, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25]

setDriftStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, 
  nBurnin::Int, currentStep::Float64) = setSmmalaDriftStep(i, acceptanceRatio,
  nMcmc, nBurnin, currentStep, driftSteps)

# Create instance malaOpts of SMMALA options
smmalaOpts = SmmalaOpts(55000, 5000, setDriftStep);

# Run SMMALA
mcmc = Array(Float64, nPars, nPars)
z = Array(Float64, nPars, nPars)
linearZvMcmc = Array(Float64, nPars, nPars)
linearCoef = Array(Float64, nPars, nPars)
quadraticZvMcmc = Array(Float64, nPars, nPars)
quadraticCoef = Array(Float64, convert(Int, nPars*(nPars+3)/2), nPars)

try
  mcmc, z = smmala(model, smmalaOpts);
  
  # Compute ZV-SMMALA mean estimators based on linear polynomial
  linearZvMcmc, linearCoef = linearZv(mcmc, z);

  # Compute ZV-SMMALA mean estimators based on quadratic polynomial
  quadraticZvMcmc, quadraticCoef = quadraticZv(mcmc, z);
catch msg
  println(msg)
end
