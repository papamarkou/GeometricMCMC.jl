function output = hmc(model, options)
% hmc: function for running HMC for a given model.

% Written by Theodore Papamarkou.
% (c) UCL, Department of Statistical Science. All rights reserved.

%% Get user options
nMcmc = options.nMcmc;
nBurnIn = options.nBurnIn;
nPosteriorSamples = nMcmc-nBurnIn;

massMatrix = options.massMatrix;
nLeap = options.nLeap;
leapStep = options.leapStep;

monitorRate = options.monitorRate;

if mod(nBurnIn, monitorRate)~=0
  error('Monitor rate must be a divisor of number of burn-in steps');
end

if mod(nMcmc, monitorRate)~=0
  error('Monitor rate must be a divisor of number of MCMC steps');
end

%% Pre-allocate memory
B = zeros(nPosteriorSamples, model.np);

%% Calculate joint log posterior for initial parameters
parameters = model.samplePrior();
logPosterior = model.logPosterior(parameters);

%% Calculate initial values of z
gradLogPosterior = model.gradLogPosterior(parameters);
Z = zeros(nPosteriorSamples, model.np);

%% Perform HMC iterations
fprintf('Initialisation completed...\n\nRunning burn-in iterations...\n');
[proposed, accepted] = deal(0);

i = 1;
while (i <= nMcmc)     
  proposedParameters = parameters;
  proposed = proposed+1;

  % Sample momentum
  proposedMomentum = (randn(1, model.np)*massMatrix)';
  
  % Calculate current H value
  hamiltonian  = -logPosterior+...
    (proposedMomentum'*(massMatrix\proposedMomentum))/2;
  
  randomSteps = ceil(rand*nLeap);
        
  % Perform leapfrog steps
  proposedGradLogPosterior = gradLogPosterior;
  for j = 1:randomSteps
    proposedMomentum = proposedMomentum+...
      (leapStep/2)*proposedGradLogPosterior;
    
    %proposedParameters = proposedParameters+...
    %  leapStep*(massMatrix\proposedMomentum);
    proposedParameters = proposedParameters+...
      leapStep*proposedMomentum;  
    
    proposedGradLogPosterior = model.gradLogPosterior(proposedParameters);
    proposedMomentum = proposedMomentum+...
      (leapStep/2)*proposedGradLogPosterior;
  end

  % Calculate proposed joint log posterior based on proposed parameters
  proposedLogPosterior = model.logPosterior(proposedParameters);
  
  % Calculate Hamiltonian based on proposed parameters
  proposedHamiltonian = -proposedLogPosterior+...
    (proposedMomentum'*(massMatrix\proposedMomentum))/2;
       
  % Accept according to ratio
  ratio = hamiltonian-proposedHamiltonian;

  if ratio > 0 || (ratio > log(rand))
    parameters = proposedParameters;
    
    logPosterior = proposedLogPosterior;
    gradLogPosterior = proposedGradLogPosterior;
    
    accepted = accepted+1;
  end

  if mod(i, monitorRate) == 0
    acceptanceRatio = accepted/proposed;
    fprintf('Iteration %u of %u: %.2f%% acceptance ratio\n', ...
      i, nMcmc, 100*acceptanceRatio);
    
    if i <= 0.1*nBurnIn
      if acceptanceRatio < 0.1
        leapStep = options.leapStep01;
      elseif 0.9 <= acceptanceRatio
        leapStep = options.leapStep02;
      end
    elseif i <= 0.15*nBurnIn
      if acceptanceRatio < 0.15
        leapStep = options.leapStep02;
      elseif 0.9 <= acceptanceRatio
        leapStep = options.leapStep03;
      end
    elseif i <= 0.2*nBurnIn
      if acceptanceRatio < 0.2
        leapStep = options.leapStep03;
      elseif 0.9 <= acceptanceRatio
        leapStep = options.leapStep04;
      end
    elseif i <= 0.25*nBurnIn
      if acceptanceRatio < 0.25
        leapStep = options.leapStep04;
      elseif 0.9 <= acceptanceRatio
        leapStep = options.leapStep05;
      end
    elseif i <= 0.3*nBurnIn
      if acceptanceRatio < 0.3
        leapStep = options.leapStep05;
      elseif 0.9 <= acceptanceRatio
        leapStep = options.leapStep06;
      end
    elseif i <= 0.35*nBurnIn
      if acceptanceRatio < 0.35
        leapStep = options.leapStep06;
      elseif 0.9 <= acceptanceRatio
        leapStep = options.leapStep07;
      end
    elseif i <= 0.4*nBurnIn
      if acceptanceRatio < 0.4
        leapStep = options.leapStep07;
      elseif 0.9 <= acceptanceRatio
        leapStep = options.leapStep08;
      end
    elseif i <= 0.45*nBurnIn
      if acceptanceRatio < 0.5
        leapStep = options.leapStep08;
      elseif 0.85 <= acceptanceRatio
        leapStep = options.leapStep09;
      end
    elseif i <= 0.5*nBurnIn
      if acceptanceRatio < 0.1
        error('Aborted: low acceptance ratio during burn-in');
      elseif acceptanceRatio < 0.5
        leapStep = options.leapStep09;
      elseif 0.8 <= acceptanceRatio
        leapStep = options.leapStep;
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
  
  % Save samples if required
  if i > nBurnIn
    B(i-nBurnIn,:) = parameters;
    Z(i-nBurnIn,:) = -gradLogPosterior/2;
  end
  
  i = i+1;
end

output = cell(1, 3);
output{1} = B;
output{2} = Z;
output{3} = [nMcmc nBurnIn nLeap leapStep];
