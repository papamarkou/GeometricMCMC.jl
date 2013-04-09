# Functions for Bayesian logit model with a Normal prior N(0, priorVar*I)

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

function tensor(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})
  p = 1./(1+exp(-data["X"]*pars))
  return ((data["X"]'.*repmat((p.*(1-p))', nPars, 1))*data["X"]
    +(eye(nPars)/data["priorVar"]))
end

function derivTensor(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})
  matrix = Array(Float64, data["nData"], nPars)
  output = Array(Float64, nPars, nPars, nPars)
  
  p = 1./(1+exp(-data["X"]*pars))
  
  for i = 1:nPars
    for j =1:nPars
      matrix[:, j] = data["X"][:, j].*((p.*(1-p)).*((1-2*p).*data["X"][:, i]))
    end
    
    output[:, :, i] = matrix'*data["X"]
  end
  
  return output
end

function randPrior(nPars::Int, data::Dict{Any, Any})
  return rand(Normal(0.0, sqrt(data["priorVar"])), nPars)
end
