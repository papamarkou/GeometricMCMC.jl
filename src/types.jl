type Model
  nPars::Int
  
  data::Union(Array{Any}, Dict{Any, Any})

  logPrior::Function
  logLikelihood::Function

  logPosterior::Function
  gradLogPosterior::Function
  tensor::Function
  derivTensor::Array{Function}

  randPrior::Function
  
  Model(nPars::Int, data::Union(Array{Any}, Dict{Any, Any}), 
    logPrior::Function, logLikelihood::Function, gradLogPosterior::Function,
    tensor::Function, derivTensor::Array{Function}, randPrior::Function) = begin
    instance = new()
    
    instance.nPars = nPars
    
    instance.data = data
    
    instance.logPrior = logPrior
    instance.logLikelihood = logLikelihood

    logPosterior(pars::Int, data::Union(Array{Any}, Dict{Any, Any})) =
      logPrior(pars, data)+logLikelihood(pars, data)
    instance.logPosterior = logPosterior
    instance.gradLogPosterior = gradLogPosterior
    instance.tensor = tensor
    for i = 1:size(derivTensor, 1)
      instance.derivTensor[i]= derivTensor[i] 
    end
    
    instance.randPrior = randPrior
  
    instance
  end

  Model(nPars::Int, data::Union(Array{Any}, Dict{Any, Any}),
    logPrior::Function, logLikelihood::Function, gradLogPosterior::Function,
    randPrior::Function) = begin
    instance = new()

    instance.nPars = nPars
        
    instance.data = data
    
    instance.logPrior = logPrior
    instance.logLikelihood = logLikelihood

    logPosterior(pars::Int, data::Union(Array{Any}, Dict{Any, Any})) =
      logPrior(pars, data)+logLikelihood(pars, data)
    instance.logPosterior = logPosterior
    instance.gradLogPosterior = gradLogPosterior

    instance.randPrior = randPrior
    
    instance
  end
end

type McmcOpts
  nMcmc::Int
  nBurnin::Int
  nPostBurnin::Int
  
  monitorRate::Int
  
  McmcOpts(nMcmc::Int, nBurnin::Int, monitorRate::Int) = begin
    instance = new()
    
    instance.n = nMcmc
    instance.nBurnin = nBurnin
    instance.nPostBurnin = instance.n-instance.nBurnin
    
    instance.monitorRate = monitorRate
    
    instance
  end
  
  McmcOpts(nMcmc::Int, nBurnin::Int) = begin
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
  
  MhOpts(nMcmc::Int, nBurnin::Int, monitorRate::Int, widthCorrection::Float64) =
  begin
    instance = new()
    
    instance.mcmc = McmcOpts(nMcmc, nBurnin, monitorRate)
    
    instance.widthCorrection = widthCorrection
    
    instance
  end
  
  MhOpts(nMcmc::Int, nBurnin::Int, widthCorrection::Float64) = begin
    instance = new()
    
    instance.mcmc = McmcOpts(nMcmc, nBurnin)
    
    instance.widthCorrection = widthCorrection
    
    instance
  end
end
