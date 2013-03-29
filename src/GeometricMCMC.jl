module GeometricMCMC
  using Base
  
  require("Distributions")
  using Distributions

  include("types.jl")
  include("mh.jl")
  include("logitNormalPrior.jl")
  include("../test/data.jl")
  
  export
    # Types
    Model
    MhOpts
    
    # Geometric MCMC functions
    mh # Metropolis-Hastings
    
    # Functions for calculating zero-variance estimators
    
    # Numeric arrays holding datasets used in the examples
    swiss
    vaso
end
