# Function for running manifold Metropolis adjusted Langevin algorithm (MMALA)
function mmala(model::Model, opts::MmalaOpts)
  mcmc = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  z = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  invGDerivG = Array(Float64, model.nPars, model.nPars, model.nPars)
  traceInvGDerivG = Array(Float64, model.nPars)
  currentSecondTerm = Array(Float64, model.nPars, model.nPars)
  proposedSecondTerm = Array(Float64, model.nPars, model.nPars)
  proposed, accepted = 0., 0.

  println("Running burn-in iterations...")

  currentPars = model.randPrior()
  currentLogPosterior = model.logPosterior(currentPars)
  currentGradLogPosterior = model.gradLogPosterior(currentPars)
  currentG = model.tensor(currentPars)
  currentInvG = inv(currentG)
  derivG = model.derivTensor(currentPars)
  currentFirstTerm = currentInvG*currentGradLogPosterior
  for i = 1:model.nPars    
    invGDerivG[:, :, i] = currentInvG*derivG[:, :, i]
    traceInvGDerivG[i] = trace(invGDerivG[:, :, i])
    currentSecondTerm[:, i] = invGDerivG[:, :, i]*currentInvG[:, i]
  end
  currentThirdTerm = currentInvG*traceInvGDerivG
  driftStep = opts.setDriftStep(1, 0., opts.mcmc.n, opts.mcmc.nBurnin, 0.)
  
  for i = 1:opts.mcmc.n
    proposed += 1
    
    proposedPars = copy(currentPars)    
    parameterMeans = (proposedPars+(driftStep/2)*currentFirstTerm
      -driftStep*sum(currentSecondTerm, 2)[:]+(driftStep/2)*currentThirdTerm) 
    cholCurrentInvG = chol(driftStep*currentInvG)
    proposedPars = parameterMeans+cholCurrentInvG'*randn(model.nPars) 
    proposedLogPosterior = model.logPosterior(proposedPars) 
    probNewGivenOld = (-sum(log(diag(cholCurrentInvG)))
      -(0.5*(parameterMeans-proposedPars)'
      *(currentG/driftStep)*(parameterMeans-proposedPars))[1])
    proposedG = model.tensor(proposedPars)
    proposedInvG = inv(proposedG)
    proposedGradLogPosterior = model.gradLogPosterior(proposedPars)
    proposedFirstTerm = proposedInvG*proposedGradLogPosterior
    derivG = model.derivTensor(proposedPars)    
    for j = 1:model.nPars               
      invGDerivG[:, :, j] = proposedInvG*derivG[:, :, j]
      traceInvGDerivG[j] = trace(invGDerivG[:, :, j])
      proposedSecondTerm[:, j] = invGDerivG[:, :, j]*proposedInvG[:, j]
    end
    proposedThirdTerm = proposedInvG*traceInvGDerivG
    parameterMeans = (proposedPars+(driftStep/2)*proposedFirstTerm
      -driftStep*sum(proposedSecondTerm, 2)[:]+(driftStep/2)*proposedThirdTerm)
    probOldGivenNew =
      (-sum(log(diag(chol(driftStep*eye(model.nPars)*proposedInvG))))
      -(0.5*(parameterMeans-currentPars)'
      *(proposedG/driftStep)*(parameterMeans-currentPars))[1])
 
    ratio =
      proposedLogPosterior+probOldGivenNew-currentLogPosterior-probNewGivenOld

    if ratio > 0 || (ratio > log(rand()))
      accepted += 1
      
      currentPars = copy(proposedPars)
      currentLogPosterior = copy(proposedLogPosterior)
      currentGradLogPosterior = copy(proposedGradLogPosterior)
      currentG = copy(proposedG)
      currentInvG = copy(proposedInvG)
      currentFirstTerm = copy(proposedFirstTerm)
      currentSecondTerm = copy(proposedSecondTerm)
      currentThirdTerm = copy(proposedThirdTerm)
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
