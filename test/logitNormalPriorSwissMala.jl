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
function setDriftStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, 
  nBurnin::Int, currentStep::Float64)
  driftStep = [1, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25]

  if i == 1
    driftStep[1]
  elseif i <= 0.1*nBurnin
    if acceptanceRatio < 0.1
      driftStep[2]
    elseif 0.9 <= acceptanceRatio
      driftStep[3]
    else
      currentStep
    end
  elseif i <= 0.15*nBurnin
    if acceptanceRatio < 0.15
      driftStep[3]
    elseif 0.9 <= acceptanceRatio
      driftStep[4]
    else
      currentStep
    end
  elseif i <= 0.2*nBurnin
    if acceptanceRatio < 0.2
      driftStep[4]
    elseif 0.9 <= acceptanceRatio
      driftStep[5]
    else
      currentStep
    end
  elseif i <= 0.25*nBurnin
    if acceptanceRatio < 0.25
      driftStep[5]
    elseif 0.9 <= acceptanceRatio
      driftStep[6]
    else
      currentStep
    end
  elseif i <= 0.3*nBurnin
    if acceptanceRatio < 0.3
      driftStep[6]
    elseif 0.9 <= acceptanceRatio
      driftStep[7]
    else
      currentStep
    end
  elseif i <= 0.35*nBurnin
    if acceptanceRatio < 0.35
      driftStep[7]
    elseif 0.9 <= acceptanceRatio
      driftStep[8]
    else
      currentStep
    end
  elseif i <= 0.4*nBurnin
    if acceptanceRatio < 0.4
      driftStep[8]
    elseif 0.9 <= acceptanceRatio
      driftStep[9]
    else
      currentStep
    end
  elseif i <= 0.45*nBurnin
    if acceptanceRatio < 0.45
      driftStep[9]
    elseif 0.85 <= acceptanceRatio
      driftStep[10]
    else
      currentStep
    end
  elseif i <= 0.5*nBurnin
    if acceptanceRatio < 0.1
      throw("Aborted: low acceptance ratio during burn-in")
    elseif acceptanceRatio < 0.5
      driftStep[10]
    elseif 0.8 <= acceptanceRatio
      driftStep[1]
    else
      currentStep
    end
  elseif i <= 0.7*nBurnin
    currentStep
  elseif i <= 0.75*nBurnin
    if acceptanceRatio < 0.1
      throw("Aborted: low acceptance ratio during burn-in")
    else
      currentStep
    end
  elseif i <= 0.9*nBurnin
    currentStep
  elseif i <= nBurnin
    if acceptanceRatio < 0.1
      throw("Aborted: low acceptance ratio during burn-in")
    elseif 0.925 <= acceptanceRatio
      throw("Aborted: high acceptance ratio during burn-in")
    else
      currentStep
    end
  # 0.5*(nBurnin+nMcmc) =nBurnin+0.5*nPostBurnin
  elseif i <= 0.5*(nBurnin+nMcmc)
    currentStep
  elseif i <= nMcmc
    if acceptanceRatio < 0.1
      throw("Aborted: low acceptance ratio during post burn-in")
    elseif 0.925 <= acceptanceRatio
      throw("Aborted: high acceptance ratio during post burn-in")
    else
      currentStep
    end
  end
end

# Create instance malaOpts of MALA options
malaOpts = MalaOpts(55000, 5000, setDriftStep);

# Run MALA
mcmc, z = mala(model, malaOpts);

# Compute ZV-MALA mean estimators based on linear polynomial
linearZvMcmc, linearCoef = linearZv(mcmc, z);

# Compute ZV-MALA mean estimators based on quadratic polynomial
quadraticZvMcmc, quadraticCoef = quadraticZv(mcmc, z);
