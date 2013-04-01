module GeometricMCMC
  using Base
  
  require("Distributions")
  using Distributions

  include("types.jl")
  include("mh.jl")
  include("mala.jl")
  #include("smmala.jl")
  #include("mmala.jl")
  #include("hmc.jl")
  #include("rmhmc.jl")
  include("linearZv.jl")
  include("quadraticZv.jl")
  include("logitNormalPrior.jl")
  include("../test/data.jl")
  
  export
    ## Types
    Model
    MhOpts
    
    ## MCMC functions
    mh # Metropolis-Hastings (MH)
    mala # Metropolis adjusted Langevin algorithm (MALA)
    smmala # Simplified Manifold MALA (SMMALA)
    mmala # Manifold MALA (MMALA)
    hmc # Hamiltonian Monte Carlo (HMC)
    rmhmc # Riemannian Manifold HMC (RMHMC)
    
    ## Functions for calculating zero-variance estimators
    linearZv # ZV-MCMC using linear polynomial
    quadraticZv # ZV-MCMC using quadratic polynomial
    
    ## Numeric arrays holding datasets used in the examples
    swiss # Swiss banknote data
    vaso # Vaso constriction data
end
