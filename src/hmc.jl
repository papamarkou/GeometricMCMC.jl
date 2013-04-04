# Function for running Hamiltonian Monte Carlo (HMC)
function hmc(model::Model, opts::HmcOpts)
  mcmc = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  z = Array(Float64, opts.mcmc.nPostBurnin, model.nPars)
  proposed, accepted = 0., 0.
  
  println("Running burn-in iterations...")

  currentPars = model.randPrior()
  currentLogPosterior = model.logPosterior(currentPars)
  gradLogPosterior = model.gradLogPosterior(currentPars)
  leapStep = opts.leapStep

  for i = 1:opts.mcmc.n
    proposed += 1  
  proposedParameters = currentPars;

  proposedMomentum = (randn(1, model.np)*opts.mass)';
 
  hamiltonian  = -currentLogPosterior+...
    (proposedMomentum'*(opts.mass\proposedMomentum))/2;
  
  randomSteps = ceil(rand*opts.nLeaps);
       
  proposedGradLogPosterior = gradLogPosterior;
  for j = 1:randomSteps
    proposedMomentum = proposedMomentum+...
      (leapStep/2)*proposedGradLogPosterior;
    
    %proposedParameters = proposedParameters+...
    %  leapStep*(opts.mass\proposedMomentum);
    proposedParameters = proposedParameters+...
      leapStep*proposedMomentum;  
    
    proposedGradLogPosterior = model.gradLogPosterior(proposedParameters);
    proposedMomentum = proposedMomentum+...
      (leapStep/2)*proposedGradLogPosterior;
  end

  proposedLogPosterior = model.logPosterior(proposedParameters);
 
  proposedHamiltonian = -proposedLogPosterior+...
    (proposedMomentum'*(opts.mass\proposedMomentum))/2;
      
  ratio = hamiltonian-proposedHamiltonian;

  if ratio > 0 || (ratio > log(rand))
    currentPars = proposedParameters;
    
    currentLogPosterior = proposedLogPosterior;
    gradLogPosterior = proposedGradLogPosterior;
    
    accepted = accepted+1;
  end

  if mod(i, monitorRate) == 0
    acceptanceRatio = accepted/proposed;
    fprintf('Iteration %u of %u: %.2f%% acceptance ratio\n', ...
      i, nMcmc, 100*acceptanceRatio);
    
    if i <= 0.1*nBurnIn
      if acceptanceRatio < 0.1
        leapStep = opts.leapStep01;
      elseif 0.9 <= acceptanceRatio
        leapStep = opts.leapStep02;
      end
    elseif i <= 0.15*nBurnIn
      if acceptanceRatio < 0.15
        leapStep = opts.leapStep02;
      elseif 0.9 <= acceptanceRatio
        leapStep = opts.leapStep03;
      end
    elseif i <= 0.2*nBurnIn
      if acceptanceRatio < 0.2
        leapStep = opts.leapStep03;
      elseif 0.9 <= acceptanceRatio
        leapStep = opts.leapStep04;
      end
    elseif i <= 0.25*nBurnIn
      if acceptanceRatio < 0.25
        leapStep = opts.leapStep04;
      elseif 0.9 <= acceptanceRatio
        leapStep = opts.leapStep05;
      end
    elseif i <= 0.3*nBurnIn
      if acceptanceRatio < 0.3
        leapStep = opts.leapStep05;
      elseif 0.9 <= acceptanceRatio
        leapStep = opts.leapStep06;
      end
    elseif i <= 0.35*nBurnIn
      if acceptanceRatio < 0.35
        leapStep = opts.leapStep06;
      elseif 0.9 <= acceptanceRatio
        leapStep = opts.leapStep07;
      end
    elseif i <= 0.4*nBurnIn
      if acceptanceRatio < 0.4
        leapStep = opts.leapStep07;
      elseif 0.9 <= acceptanceRatio
        leapStep = opts.leapStep08;
      end
    elseif i <= 0.45*nBurnIn
      if acceptanceRatio < 0.5
        leapStep = opts.leapStep08;
      elseif 0.85 <= acceptanceRatio
        leapStep = opts.leapStep09;
      end
    elseif i <= 0.5*nBurnIn
      if acceptanceRatio < 0.1
        error('Aborted: low acceptance ratio during burn-in');
      elseif acceptanceRatio < 0.5
        leapStep = opts.leapStep09;
      elseif 0.8 <= acceptanceRatio
        leapStep = opts.leapStep;
      end
    elseif (0.7*nBurnIn <= i) && (i < 0.75*nBurnIn)
      if acceptanceRatio < 0.1
        error('Aborted: low acceptance ratio during burn-in');
      end
    elseif (0.9*nBurnIn <= i) && (i <= nBurnIn)
      if acceptanceRatio < 0.1
        error('Aborted: low acceptance ratio during burn-in');
      elseif 0.925 <= acceptanceRatio
        error('Aborted: high acceptance ratio during burn-in');
      end
      if i == nBurnIn
        fprintf('Burn-in completed...\n\nRunning post burn-in MCMC...\n');
      end
    elseif (nBurnIn+0.5*nPosteriorSamples <= i) && (i <= nMcmc)
      if acceptanceRatio < 0.1
        error('Aborted: low acceptance ratio during post burn-in');
      elseif 0.925 <= acceptanceRatio
        error('Aborted: high acceptance ratio during post burn-in');
      end
    end
    
    [proposed, accepted] = deal(0);
  end
 
  if i > nBurnIn
    mcmc(i-nBurnIn,:) = currentPars;
    z(i-nBurnIn,:) = -gradLogPosterior/2;
  end
end

  return mcmc, z
end
