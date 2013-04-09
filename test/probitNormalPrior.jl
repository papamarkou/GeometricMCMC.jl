# Functions for Bayesian probit model with a Normal prior N(0, priorVar*I)

function logPrior(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})
  return (-dot(pars,pars)/data["priorVar"]
    -nPars*log(2*pi*data["priorVar"]))/2
end

function logLikelihood(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})
  XPars = data["X"]*pars
  return (dot(logOfNormalCdf(XPars), data["y"])
    +dot(logOfNormalCdf(-XPars), 1-data["y"]))
end

function gradLogPosterior(pars::Vector{Float64}, nPars::Int,
  data::Dict{Any, Any})    
  XPars = data["X"]*pars  
  return (data["X"]'
    *(data["y"].*exp(-(XPars.^2+log(2*pi))/2-logOfNormalCdf(XPars))
    -(1-data["y"]).*exp(-(XPars.^2+log(2*pi))/2-logOfNormalCdf(-XPars)))
    -pars/data["priorVar"])
end

function tensor(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})    
  XPars = data["X"]*pars
  vector =
    exp(-XPars.^2-logOfNormalCdf(XPars)-logOfNormalCdf(-XPars)-log(2*pi))
  return ((data["X"]'.*repmat(vector', nPars, 1))*data["X"]
    +(eye(nPars)/data["priorVar"]))
end

function derivTensor(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})
  output = Array(Float64, nPars, nPars, nPars)
  
  XPars = data["X"]*pars
  vector01 =
    exp(-XPars.^2-2*logOfNormalCdf(XPars)-logOfNormalCdf(-XPars)-log(2*pi))

  for i = 1:nPars
    vector02 = (vector01.*(exp(-(XPars.^2+log(2*pi))/2-logOfNormalCdf(-XPars))
      -2*(pdf(Normal(0., 1.), XPars)
      +XPars.*cdf(Normal(0., 1.), XPars))).*data["X"][:, i])
    
    output[:, :, i] = (data["X"]'.*repmat(vector02', nPars, 1))*data["X"]
  end
  
  return output
end

function randPrior(nPars::Int, data::Dict{Any, Any})
  return rand(Normal(0.0, sqrt(data["priorVar"])), nPars)
end
