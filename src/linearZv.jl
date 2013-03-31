# Function for calculating ZV-MCMC estimators using linear polynomial
function linearZv(mcmc::Array{Float64, 2}, z::Array{Float64, 2})
  nPars = size(mcmc, 2)

  covAll = Array(Float64, nPars+1, nPars+1, nPars)
  precision = Array(Float64, nPars, nPars, nPars)
  sigma = Array(Float64, nPars, nPars)
  a = Array(Float64, nPars, nPars)

  for i = 1:nPars
     covAll[:, :, i] = cov([z mcmc[:, i]])
     precision[:, :, i] = inv(covAll[1:nPars, 1:nPars, i])
     sigma[:, i] = covAll[1:nPars, nPars+1, i]
     a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  zvMcmc = mcmc+z*a
  
  return zvMcmc, a
end
