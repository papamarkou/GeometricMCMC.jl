# Function for running simplified manifold Metropolis adjusted Langevin 
# algorithm (SMMALA)
function smmala(model::Model, opts::MalaOpts)
  mcmc = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  z = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  proposed, accepted = 0., 0.
  
  println("Running burn-in iterations...")

  currentPars = model.randPrior()
  currentLogPosterior = model.logPosterior(currentPars)
  currentGradLogPosterior = model.gradLogPosterior(currentPars)
  currentG = model.tensor(currentPars)
  currentInvG = inv(currentG)
  currentFirstTerm = currentInvG*currentGradLogPosterior
  driftStep = opts.setDriftStep(1, 0., opts.mcmc.n, opts.mcmc.nBurnin, 0.)
  
  for i = 1:opts.mcmc.n
    proposed += 1
    
    proposedPars = copy(currentPars) 
    parameterMeans = proposedPars+(driftStep/2)*currentFirstTerm
    cholCurrentInvG = chol(driftStep*currentInvG)
    proposedPars = parameterMeans+cholCurrentInvG'*randn(model.nPars)
    proposedLogPosterior = model.logPosterior(proposedPars)    
    probNewGivenOld = (-sum(log(diag(cholCurrentInvG)))
      -0.5*(parameterMeans-proposedPars)'
      *(currentG/driftStep)*(parameterMeans-proposedPars))
    proposedG = model.tensor(proposedPars)
    proposedInvG = inv(proposedG)
    proposedGradLogPosterior = model.gradLogPosterior(proposedPars)
    proposedFirstTerm = proposedG\proposedGradLogPosterior
    parameterMeans = proposedPars+(driftStep/2)*proposedFirstTerm
    probOldGivenNew = 
      (-sum(log(diag(chol(driftStep*eye(model.nPars)/proposedG))))
      -0.5*(parameterMeans-currentPars)'
      *(proposedG/driftStep)*(parameterMeans-currentPars))
   
   ratio =
     proposedLogPosterior+probOldGivenNew-currentLogPosterior-probNewGivenOld
   
    if ratio[1] > 0 || (ratio[1] > log(rand()))
      accepted += 1
      
      currentPars = copy(proposedPars)
      currentLogPosterior = copy(proposedLogPosterior)
      currentGradLogPosterior = copy(proposedGradLogPosterior)  
      currentG = copy(proposedG)
      currentInvG = copy(proposedInvG)    
      currentFirstTerm  = copy(proposedFirstTerm)
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
