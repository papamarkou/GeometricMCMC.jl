# Function for calculating ZV-MCMC estimators using quadratic polynomial
function quadraticZv(mcmc::Array{Float64, 2}, z::Array{Float64, 2})
  nData, nPars = size(mcmc)
  k = convert(Int, nPars*(nPars+3)/2)
  l = 2*nPars+1
  
  zQuadratic = Array(Float64, nData, k)  
  covAll = Array(Float64, k+1, k+1, nPars)
  precision = Array(Float64, k, k, nPars)
  sigma = Array(Float64, k, nPars)
  a = Array(Float64, k, nPars)

  zQuadratic[:, 1:nPars] = z
  zQuadratic[:, (nPars+1):(2*nPars)] = 2*z.*mcmc-1
  for i = 1:(nPars-1)
    for j = (i+1):nPars
      zQuadratic[:, l] = mcmc[:, i].*z[:, j]+mcmc[:, j].*z[:, i]
      l += 1
    end
  end

  for i = 1:nPars
    covAll[:, :, i] = cov([zQuadratic mcmc[:, i]]);
    precision[:, :, i] =
      inv(covAll[1:k, 1:k, i])
    sigma[:, i] = covAll[1:k, k+1, i]
    a[:, i] = -precision[:, :, i]*sigma[:, i]
  end

  zvMcmc = mcmc+zQuadratic*a;

  return zvMcmc, a
end
