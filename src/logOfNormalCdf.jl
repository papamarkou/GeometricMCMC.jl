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
