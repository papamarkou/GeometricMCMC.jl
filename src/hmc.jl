# Function for running Hamiltonian Monte Carlo (HMC)
function hmc(model::Model, opts::HmcOpts)
  mcmc = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  z = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  proposed, accepted = 0., 0.
  
  println("Running burn-in iterations...")

  currentPars = model.randPrior()
  currentLogPosterior = model.logPosterior(currentPars)
  currentGradLogPosterior = model.gradLogPosterior(currentPars)
  invMass = inv(opts.mass)
  leapStep = opts.setLeapStep(1, 0., opts.mcmc.n, opts.mcmc.nBurnin, 0.)

  for i = 1:opts.mcmc.n
    proposed += 1 
    
    proposedPars = copy(currentPars)
    momentum = opts.mass'*randn(model.nPars)
    currentHamiltonian = -currentLogPosterior+(momentum'*(invMass*momentum))/2
    randomSteps = ceil(rand()*opts.nLeaps)
    proposedGradLogPosterior = copy(currentGradLogPosterior)
    for j = 1:randomSteps
      momentum = momentum+(leapStep/2)*proposedGradLogPosterior
      proposedPars = proposedPars+leapStep*momentum
      proposedGradLogPosterior = model.gradLogPosterior(proposedPars)
      momentum = momentum+(leapStep/2)*proposedGradLogPosterior
    end
    proposedLogPosterior = model.logPosterior(proposedPars)
    proposedHamiltonian = -proposedLogPosterior+(momentum'*(invMass*momentum))/2
   
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
