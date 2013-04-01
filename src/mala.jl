# Function for running Metropolis adjusted Langevin algorithm (MALA)
function mala(model::Model, opts::MalaOpts)
  mcmc = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  z = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  proposed, accepted, acceptanceRatio = 0., 0., 0.
  
  println("Running burn-in iterations...")  

  currentPars = model.randPrior()
  currentLogPosterior = model.logPosterior(currentPars)
  currentGradLogPosterior = model.gradLogPosterior(currentPars)
  driftStep =
    opts.setDriftStep(1, acceptanceRatio, opts.mcmc.n, opts.mcmc.nBurnin, 0.)

  for i = 1:opts.mcmc.n
    proposed += 1
    
    proposedPars = copy(currentPars)
    proposedGradLogPosterior = model.gradLogPosterior(proposedPars)
    parameterMeans = proposedPars+(driftStep/2)*proposedGradLogPosterior
    proposedPars = parameterMeans+sqrt(driftStep)*randn(model.nPars)
    proposedLogPosterior = model.logPosterior(proposedPars)
    probNewGivenOld =
      sum(-(parameterMeans-proposedPars).^2/(2*driftStep)-log(2*pi*driftStep)/2)
    proposedGradLogPosterior = model.gradLogPosterior(proposedPars)
    parameterMeans = proposedPars+ (driftStep/2)*proposedGradLogPosterior
    probOldGivenNew =
      sum(-(parameterMeans-currentPars).^2/(2*driftStep)-log(2*pi*driftStep)/2)
     
    ratio =
      proposedLogPosterior+probOldGivenNew-currentLogPosterior-probNewGivenOld
             
    if ratio > 0 || (ratio > log(rand()))
      accepted += 1
      
      currentPars = copy(proposedPars)
      currentLogPosterior = copy(proposedLogPosterior)
      currentGradLogPosterior = copy(proposedGradLogPosterior)
    end

    if i > opts.mcmc.nBurnin
      mcmc[i-opts.mcmc.nBurnin, :] = currentPars;
      z[i-opts.mcmc.nBurnin, :] = -currentGradLogPosterior/2;
    end
    
    if mod(i, opts.mcmc.monitorRate) == 0
      acceptanceRatio = accepted/proposed
     
    driftStep = opts.setDriftStep(i, acceptanceRatio, opts.mcmc.n,
      opts.mcmc.nBurnin, driftStep)
      
      println("Iteration $i of $(opts.mcmc.n): ", round(100*acceptanceRatio, 2),
        " % acceptance ratio")
    
      proposed, accepted = 0., 0.
    end
    
    if i == opts.mcmc.nBurnin
      println("Burn-in completed...\n\nRunning post burn-in MCMC...");
    end
  end

  return mcmc, z
end
