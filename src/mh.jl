function mh(model::Model, opts::MhOptions)
  mcmc = Array(opts.mcmc.nPostBurnin, model.nPars)

  parameters = model.samplePrior();
  logPosterior = model.logPosterior(parameters);

%% Calculate initial values of z
gradLogPosterior = model.gradLogPosterior(parameters);
Z = zeros(nPosteriorSamples, model.np);

%% Perform Metropolis-Hastings iterations
fprintf('Initialisation completed...\n\nRunning burn-in iterations...\n');
[proposed, accepted, acceptanceRatio] = deal(zeros(model.np, 1));
proposalSd = ones(model.np, 1);

for i = 1:nMcmc
  for j = 1:model.np
    proposedParameters = parameters;
    proposedParameters(j) = proposedParameters(j)+randn*proposalSd(j);
    
    proposed(j) = proposed(j)+1;
    
    % Calculate proposed joint log posterior based on proposed parameters
    proposedLogPosterior = model.logPosterior(proposedParameters);
    
    % Calculate proposed grad of log posterior based on proposed parameters
    proposedGradLogPosterior = model.gradLogPosterior(proposedParameters);
    
    % Accept according to ratio
    ratio = proposedLogPosterior-logPosterior;
        
    if ratio > 0 || (ratio > log(rand))
      parameters = proposedParameters;
            
      logPosterior = proposedLogPosterior;
      gradLogPosterior = proposedGradLogPosterior;
      
      accepted(j) = accepted(j)+1;
    end        
  end
  
  % Save samples if required
  if i > nBurnIn
    mcmc(i-nBurnIn, :) = parameters;
    Z(i-nBurnIn, :) = -gradLogPosterior/2;
  end
  
  if mod(i, monitorRate) == 0          
    % Adjust proposal width during burn-in phase
    if i < nBurnIn
      for j = 1:model.np
        acceptanceRatio(j) = accepted(j)/proposed(j);
            
        if acceptanceRatio(j) > 0.6
          proposalSd(j) = proposalSd(j)*(1+proposalWitdthCorrection);
        elseif acceptanceRatio(j) < 0.2
          proposalSd(j) = proposalSd(j)*(1-proposalWitdthCorrection);
        end        
      end
    end
    
    fprintf('Iteration %u of %u:\n', i, nMcmc);
    for j = 1:model.np
      fprintf('  Parameter %u: %.2f%% acceptance ratio\n', ...
        j, 100*accepted(j)/proposed(j));
    end
    [proposed, accepted] = deal(zeros(model.np, 1));
  end

  if i == nBurnIn
    fprintf('Burn-in completed...\n\nRunning post burn-in MCMC...\n');
  end
end

output = cell(1, 3);
output{1} = mcmc;
output{2} = Z;
output{3} = [nMcmc nBurnIn proposalWitdthCorrection];
end
