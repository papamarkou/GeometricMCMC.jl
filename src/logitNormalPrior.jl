# Functions for Bayesian logit model with a normal Prior N(0, aI)

function logPrior(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})
  return (-dot(pars,pars)/data["priorVar"]
    -nPars*log(2*pi*data["priorVar"]))/2
end

function logLikelihood(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})
  XPars = data["X"]*pars
  return (XPars'*data["y"]-sum(log(1+exp(XPars))))[1]
end

function gradLogPosterior(pars::Vector{Float64}, nPars::Int,
  data::Dict{Any, Any})
  return (data["X"]'*(data["y"]-1./(1+exp(-data["X"]*pars)))
    -pars/data["priorVar"])
end

function randPrior(nPars::Int, data::Dict{Any, Any})
  return rand(Normal(0.0, sqrt(data["priorVar"])), nPars)
end
