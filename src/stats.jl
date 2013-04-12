# Function for calculating the sample autocovariance of a vector for lags from 
# 0 to maxlag, returning a vector of length maxlag+1
function autocov(sample::Vector{Float64}, maxLag::Int)
  n = size(sample, 1)

  if n <= maxLag
    throw("The length of the input vector must be at least maxlag+1")
  end

  sample = sample-mean(sample)

  acv = Array(Float64, maxLag+1)

  for i = 0:maxLag
     acv[i+1] = dot(sample[1:n-i], sample[1+i:n])
  end

  acv = acv/n
  
  return acv
end

# Function for calculating the standard Normal cumulative density function more 
# accurately for small values of its argument
function logOfNormalCdf(x::Union(Float64, Vector{Float64}))

  c::Float64 = -6.5
  a = [1, -1, 3, -15, 105, -945, 10395, -135135, 2027025, -34459425, 654729075]
  
  n = size(x, 1)
  y = Array(Float64, n)
  
  for i in 1:n
    if x[i]>c
      y[i] = log(cdf(Normal(0., 1.), x[i]))
    else
      y[i] = log(-dot(a, x[i].^(-2*(1:11)+1)))-x[i]^2/2-log(2*pi)/2
    end
  end
  
  return y
end

# Function for estimating the variance of a single MCMC chain using the initial 
# monotone sequence estimator of Geyer, see
# Practical Markov Chain Monte Carlo, Charles J. Geyer, Statistical Science,
# Vol. 7, No. 4. (1992), pp. 473-483, doi:10.2307/2246094
function imse(sample::Union(Array{Float64, 1}, Array{Float64, 2}), maxLag::Int)
  nData = size(sample, 1)
  nPars = size(sample, 2)  

  k = convert(Int, floor((maxLag-1)/2))
  
  # Preallocate memory
  acv = Array(Float64, maxLag+1, nPars)
  g = Array(Float64, k+1, nPars)
  m = (k+1)*ones(nPars)
  variance = Array(Float64, nPars)

  # Calculate empirical autocovariance
  for i = 1:nPars
    acv[:, i] = autocov(sample[:, i], maxLag)
  end

  # Calculate \hat{G}_{n, m} from the autocovariances, see pp. 477 in Geyer
  for i = 1:nPars
    for j = 0:k
      g[j+1, i] = acv[2*j+1, i]+acv[2*j+2, i]
      if g[j+1, i] <= 0
        m[i] = j
        break
      end
    end
  end

  # Create the monotone sequence of g
  for i = 1:nPars
    if m[i] > 1
      for j = 2:m[i]
        if g[j, i] > g[j-1, i]
          g[j, i] = g[j-1, i]
        end
      end
    end
  end

  # Calculate the initial monotone sequence estimator
  for i = 1:nPars
    variance[i] = (-acv[1, i]+2*sum(g[1:m[i], i]))/nData
  end
  
  return variance
end
