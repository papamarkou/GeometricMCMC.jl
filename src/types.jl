type Model
  nPars::Uint
  
  data::Array{Any}

  logPrior::Function
  logLikelihood::Function
  
  logPosterior::Function
  gradLogPosterior::Function
  tensor::Function
  derivTensor::Array{Function}

  randPrior::Function
  
  Model(nPars::Uint, data::Array{Any}, logPrior::Function, 
    logLikelihood::Function, gradLogPosterior::Function, tensor::Function, 
    derivTensor::Array{Function}, randPrior::Function) = begin
    instance = new()
    
    instance.nPars = nPars
    
    instance.data = data
    
    instance.logPrior = logPrior
    instance.logLikelihood = logLikelihood
    
    instance.logPosterior = instance.logLikelihood+instance.logPrior
    instance.gradLogPosterior = gradLogPosterior
    instance.tensor = tensor
    for i = 1:size(derivTensor, 1)
      instance.derivTensor[i]= derivTensor[i] 
    end
  
    instance
  end

  Model(nPars::Uint, data::Array{Any}, logPrior::Function, 
    logLikelihood::Function, gradLogPosterior::Function, randPrior::Function) =
    begin
    instance = new()

    instance.nPars = nPars
        
    instance.data = data
    
    instance.logPrior = logPrior
    instance.logLikelihood = logLikelihood

    instance.logPosterior = instance.logLikelihood+instance.logPrior
    instance.gradLogPosterior = gradLogPosterior
  
    instance
  end
end

type McmcOpts
  nMcmc::Uint
  nBurnin::Uint
  nPostBurnin::Uint
  
  monitorRate::Uint
  
  McmcOpts(nMcmc::Uint, nBurnin::Uint, monitorRate::Uint) = begin
    instance = new()
    
    instance.n = nMcmc
    instance.nBurnin = nBurnin
    instance.nPostBurnin = instance.n-instance.nBurnin
    
    instance.monitorRate = monitorRate
    
    instance
  end
  
  McmcOpts(nMcmc::Uint, nBurnin::Uint) = begin
    instance = new()
    
    instance.n = nMcmc
    instance.nBurnin = nBurnin
    instance.nPostBurnin = instance.n-instance.nBurnin
    
    instance.monitorRate = 100
    
    instance
  end
end

type MhOpts
  mcmc::McmcOpts
 
  widthCorrection::Float64
  
  MhOpts(nMcmc::Uint, nBurnin::Uint, monitorRate::Uint, 
    widthCorrection::Float64) = begin
    instance = new()
    
    instance.mcmc = McmcOpts(nMcmc, nBurnin, monitorRate)
    
    instance.widthCorrection = widthCorrection
    
    instance
  end
  
  MhOpts(nMcmc::Uint, nBurnin::Uint, widthCorrection::Float64) = begin
    instance = new()
    
    instance.mcmc = McmcOpts(nMcmc, nBurnin)
    
    instance.widthCorrection = widthCorrection
    
    instance
  end
end
