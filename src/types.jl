type Model
  nPars::Int
  
  data::Union(Array{Any}, Dict{Any, Any})

  logPrior::Function
  logLikelihood::Function

  logPosterior::Function
  gradLogPosterior::Function
  tensor::Function
  derivTensor::Function

  randPrior::Function
  
  Model(nPars::Int, data::Union(Array{Any}, Dict{Any, Any}), 
    logPrior::Function, logLikelihood::Function, gradLogPosterior::Function,
    tensor::Function, derivTensor::Function, randPrior::Function) = begin
    instance = new()
    
    instance.nPars = nPars
    
    instance.data = data
   
    instance.logPrior = (pars::Vector{Float64} -> logPrior(pars, nPars, data))
    instance.logLikelihood =
      (pars::Vector{Float64} -> logLikelihood(pars, nPars, data))

    instance.logPosterior = (pars::Vector{Float64} ->
      logPrior(pars, nPars, data)+logLikelihood(pars, nPars, data))
    instance.gradLogPosterior =
      (pars::Vector{Float64} -> gradLogPosterior(pars, nPars, data))
    instance.tensor = (pars::Vector{Float64} -> tensor(pars, nPars, data))
    instance.derivTensor=
      (pars::Vector{Float64} -> derivTensor(pars, nPars, data))
   
    instance.randPrior = (() -> randPrior(nPars, data))
  
    instance
  end

  Model(nPars::Int, data::Union(Array{Any}, Dict{Any, Any}),
    logPrior::Function, logLikelihood::Function, gradLogPosterior::Function,
    randPrior::Function) = begin
    instance = new()

    instance.nPars = nPars
        
    instance.data = data
    
    instance.logPrior = (pars::Vector{Float64} -> logPrior(pars, nPars, data))
    instance.logLikelihood =
      (pars::Vector{Float64} -> logLikelihood(pars, nPars, data))

    instance.logPosterior = (pars::Vector{Float64} ->
      logPrior(pars, nPars, data)+logLikelihood(pars, nPars, data))
    instance.gradLogPosterior =
      (pars::Vector{Float64} -> gradLogPosterior(pars, nPars, data))
   
    instance.randPrior = (() -> randPrior(nPars, data))
    
    instance
  end
end

type McmcOpts
  n::Int
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

type MalaOpts
  mcmc::McmcOpts

  setDriftStep::Function
  
  MalaOpts(nMcmc::Int, nBurnin::Int, monitorRate::Int, driftStep::Float64) =
  begin
    instance = new()
    
    instance.mcmc = McmcOpts(nMcmc, nBurnin, monitorRate)
    
    instance.setDriftStep = ((currentIter::Int, acceptanceRatio::Float64, 
      nMcmc::Int, nBurnin::Int, currentStep::Float64) -> driftStep)
    
    instance
  end
  
  MalaOpts(nMcmc::Int, nBurnin::Int, driftStep::Float64) = begin
    instance = new()
    
    instance.mcmc = McmcOpts(nMcmc, nBurnin)
   
    instance.setDriftStep = ((currentIter::Int, acceptanceRatio::Float64, 
      nMcmc::Int, nBurnin::Int, currentStep::Float64) -> driftStep)
    
    instance
  end
  
  MalaOpts(nMcmc::Int, nBurnin::Int, monitorRate::Int, setDriftStep::Function) =
  begin
    instance = new()
    
    instance.mcmc = McmcOpts(nMcmc, nBurnin, monitorRate)
    
    instance.setDriftStep = ((currentIter::Int, acceptanceRatio::Float64, 
      nMcmc::Int, nBurnin::Int, currentStep::Float64) -> 
      setDriftStep(currentIter, acceptanceRatio, nMcmc, nBurnin, currentStep))
    
    instance
  end
  
  MalaOpts(nMcmc::Int, nBurnin::Int, setDriftStep::Function) = begin
    instance = new()
    
    instance.mcmc = McmcOpts(nMcmc, nBurnin)
    
    instance.setDriftStep = ((currentIter::Int, acceptanceRatio::Float64, 
      nMcmc::Int, nBurnin::Int, currentStep::Float64) -> 
      setDriftStep(currentIter, acceptanceRatio, nMcmc, nBurnin, currentStep))
    
    instance
  end
end
