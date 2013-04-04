# Function for running Riemannian manifold Hamiltonian Monte Carlo (RMHMC)
function rmhmc(model::Model, opts::RmhmcOpts)
  mcmc = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  z = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  proposedGradLogPosterior = Array(Float64, model.nPars)
  invGDerivG = Array(Float64, model.nPars, model.nPars, model.nPars)
  traceInvGDerivG = Array(Float64, model.nPars)
  momentumTerm = Array(Float64, model.nPars)
  proposed, accepted = 0., 0.
  
  println("Running burn-in iterations...")

  currentPars = model.randPrior()
  currentLogPosterior = model.logPosterior(currentPars)
  currentGradLogPosterior = model.gradLogPosterior(currentPars)
  leapStep = opts.setLeapStep(1, 0., opts.mcmc.n, opts.mcmc.nBurnin, 0.)
  
  for i = 1:opts.mcmc.n
    proposed += 1
 
    proposedPars = copy(currentPars)
    G = model.tensor(proposedPars)
    invG = inv(G)
    cholG = chol(G)
    momentum = cholG'*randn(model.nPars)
    currentLogDetG = 0.5*(log(2)+model.nPars*log(pi)+2*sum(log(diag(cholG))))
    currentHamiltonian  = 
      -currentLogPosterior+currentLogDetG+(momentum'*(invG*momentum))/2
    derivG = model.derivTensor(proposedPars)
    for j = 1:model.nPars   
      invGDerivG[:, :, j] = invG*derivG[:, :, j]
      traceInvGDerivG[j] = trace(invGDerivG[:, :, j])
    end    
    timeStep = (randn() > 0.5 ? 1. : -1.)
    randomSteps = ceil(rand()*opts.nLeaps)
    for j = 1:randomSteps
      leapGradLogPosterior  = model.gradLogPosterior(proposedPars)
      leapMomentum = copy(momentum)
      for k = 1:opts.nNewton
        invGMomentum = invG*leapMomentum
        for r = 1:model.nPars
          momentumTerm[r] = 
            (0.5*(leapMomentum'*invGDerivG[:, :, r]*invGMomentum))[1]
        end
        leapMomentum = (momentum
          +timeStep*(leapStep/2)*(leapGradLogPosterior-0.5*traceInvGDerivG
          +momentumTerm))
      end
      momentum = copy(leapMomentum)
      leapInvGMomentum = invG*momentum
      leapParameters = copy(proposedPars)
      for k = 1:opts.nNewton
        G = model.tensor(leapParameters)
        invGMomentum = G\momentum
        leapParameters =
          proposedPars+(timeStep*(leapStep/2))*(leapInvGMomentum+invGMomentum)
      end
      proposedPars = copy(leapParameters)
      G = model.tensor(proposedPars)
      invG = inv(G)
      derivG = model.derivTensor(proposedPars)
      for k = 1:model.nPars
        invGDerivG[:, :, k] = invG*derivG[:, :, k]
        traceInvGDerivG[k] = trace(invGDerivG[:, :, k])
      end
      invGMomentum = invG*momentum
      for k = 1:model.nPars
        momentumTerm[k] = (0.5*(momentum'*invGDerivG[:, :, k]*invGMomentum))[1]
      end
      proposedGradLogPosterior = model.gradLogPosterior(proposedPars)
      momentum = (momentum  
        +timeStep*(leapStep/2)*(proposedGradLogPosterior-0.5*traceInvGDerivG
        +momentumTerm))
    end
    proposedLogPosterior = model.logPosterior(proposedPars)
    proposedLogDet = 0.5*(log(2)+model.nPars*log(pi)+2*sum(log(diag(chol(G)))))
    proposedHamiltonian =
      -proposedLogPosterior+proposedLogDet+(momentum'*(invG*momentum))/2
      
    ratio = (currentHamiltonian-proposedHamiltonian)[1]
          
    if ratio > 0 || (ratio > log(rand()))
      accepted += 1
      
      currentPars = copy(proposedPars)
      currentLogPosterior = copy(proposedLogPosterior)
      currentGradLogPosterior = copy(proposedGradLogPosterior)
    end

    if i > opts.mcmc.nBurnin
      mcmc[i-opts.mcmc.nBurnin, :] = currentPars
      z[i-opts.mcmc.nBurnin, :] = -currentGradLogPosterior/2
    end
    
    if mod(i, opts.mcmc.monitorRate) == 0
      acceptanceRatio = accepted/proposed
     
    leapStep = opts.setLeapStep(i, acceptanceRatio, opts.mcmc.n,
      opts.mcmc.nBurnin, leapStep)
      
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
