module GeometricMCMC
  using Base
  
  require("Distributions")
  using Distributions

  include("types.jl")
  include("setStep.jl")
  include("mh.jl")
  include("mala.jl")
  include("smmala.jl")
  include("mmala.jl")
  include("hmc.jl")
  include("rmhmc.jl")
  include("linearZv.jl")
  include("quadraticZv.jl")
  include("logitNormalPrior.jl")
  
  export
    ## Types
    Model,
    MhOpts,
    MalaOpts,
    SmmalaOpts,
    MmalaOpts,
    HmcOpts,
    RmhmcOpts,
    
    ## Auxiliary step functions
    setStep, # Generic step function
    setMalaDriftStep, # Function for adjusting MALA drift step
    setSmmalaDriftStep, # Function for adjusting SMMALA drift step
    setMmalaDriftStep, # Function for adjusting MMALA drift step
    setHmcLeapStep, # Function for adjusting HMC leap step
    setRmhmcLeapStep, # Function for adjusting RMHMC leap step
    
    ## MCMC functions
    mh, # Metropolis-Hastings (MH)
    mala, # Metropolis adjusted Langevin algorithm (MALA)
    smmala, # Simplified Manifold MALA (SMMALA)
    mmala, # Manifold MALA (MMALA)
    hmc, # Hamiltonian Monte Carlo (HMC)
    rmhmc, # Riemannian manifold HMC (RMHMC)
    
    ## Functions for calculating zero-variance estimators
    linearZv, # ZV-MCMC using linear polynomial
    quadraticZv # ZV-MCMC using quadratic polynomial
    
    ### Numeric arrays holding datasets used in the examples
    #swiss, # Swiss banknote data
    #vaso # Vaso constriction data
end
