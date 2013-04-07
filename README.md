## Overview of package's scope

This package implements the following 6 MCMC algorithms:

<table>
  <tr>
    <th></th><th>Julia function</th><th>MCMC algorithm</th>
  </tr>
  <tr>
    <td>1</td><td>mh</td><td>Metropolis-Hastings (MH)</td>
  </tr>
  <tr>
    <td>2</td><td>mala</td><td>Metropolis adjusted Langevin algorithm (MALA)</td>
  </tr>
  <tr>
    <td>3</td><td>smmala</td><td>Simplified Manifold MALA (SMMALA)</td>
  </tr>
  <tr>
    <td>4</td><td>mmala</td><td>Manifold MALA (MMALA)</td>
  </tr>
  <tr>
    <td>5</td><td>hmc</td><td>Hamiltonian Monte Carlo (HMC)</td>
  </tr>
  <tr>
    <td>6</td><td>rmhmc</td><td>Riemannian manifold HMC (RMHMC)</td>
  </tr>
</table>

More details for the geometric MCMC methods of the package can be found in [this article](http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00765.x/full).

Furthermore, the package provides the `linearZV()` and `quadraticZV()` functions for the computation of the zero-variance (ZV) Monte Carlo Bayesian estimators, see [this publication](http://link.springer.com/article/10.1007%2Fs11222-012-9344-6).

## Tutorial

This file serves as a tutorial explaining how to use the MCMC routines of the 
package, as well as the `linearZV()` and `quadraticZV()` functions. Code for 
the logit model with a Normal prior is provided to demonstrate usage and to 
follow through the tutorial.

To invoke each of the MCMC methods of the package, it is required to provide 
two input arguments. The first argument is an instance of the `Model` type, 
defined in the package, and is common across all 6 MCMC routines. The second 
argument is an instance of the algorithm's options type and is specific to the 
algorithm.

### The `Model` type

The `Model` type provides the statistical model to the MCMC routines. This 
includes the functions defining the model, the number of the model's parameters 
and the data.

More specifically, the functions required for defining the model are the 
log-prior, the log-likelihood, the gradient of the log-posterior, the 
position-specific metric tensor of the Riemannian manifold of the parameter 
space and the tensor's partial derivatives with respect to each of the 
parameters. These functions need to be known in closed form as the package 
stands so far. The log-posterior is also one of the model's functions and it 
does not need to be specified by the user, since the `Model` type sets it to be 
the sum of the log-likelihood with the log-prior.

It is apparent that the `Model` type represents a Bayesian model. However, it 
is also possible to accommodate simpler statistical models, such a non-Bayesian 
log-target. This can be achieved, for instance, by setting the log-likelihood 
to be equal to the log-target and the improper log-prior to be zero.

For ease of use, all the user-defined functions in the Model type (apart 
from the ones starting with "rand" as explained later in the tutorial) 
conventionally share the same signature

    function myFunction(pars::Vector{Float64}, nPars::Int, data::Union(Array{Any}, Dict{Any, Any}))

where `pars` are the model's parameters simulated by the MCMC algorithm and 
thus not needed to be numerically specified by the user, `nPars` is the number 
of parameters and `data` is an Array `Array{Any}` or a dictionary `Dict{Any, 
Any}` holding the data.

The Model can be instantiated with fewer arguments. For instance, the
Metropolis-Hastings function `mh` requires only the log-prior, log-likelihood 
and the gradient of the log-posterior. In fact, the gradient of the 
log-posterior is not necessary for running a Metropolis-Hastings MCMC 
simulation. Nevertheless, it has been set as a required argument so that `mh` 
returns the zero-variance control variates along with the MCMC output.
Similarly, the log-prior, log-likelihood and the gradient of the log-posterior 
suffice as arguments in the instantiation of `Model` in order to run MALA or 
HMC. SMMALA requires additionally the metric tensor. The partial derivatives of 
the metric tensor with respect to the parameters are needed in the `Model` 
instantiation only for MMALA or RMHMC simulations.

### An example: the Bayesian logit model with Normal prior

#### Creating the `data` member of the `Model` type

To illustrate how to create a Bayesian `Model`, the logit regression with a 
Normal prior is considered. The data are then the design matrix `X`, with 
`nData` rows and `nPars` columns, and the binary esponse variable `y`, which is 
a vector with `nData` elements. `nData` and`nPars` are the number of data 
points and the number of parameters respectively. Assuming a standard normal 
prior N(0, priorVar*I), the prior's variance `priorVar` is also part of the 
data, in the broad sense of the word.

It is up to the user's preference whether to define `data` as `Array{Any}` or a 
dictionary `Dict{Any, Any}`, as long as `data` is manipulated accordingly in 
the subsequent definition of the `Model` functions. In what follows, `data` is 
introduced as a dictionary, holding the design matrix `X`, the response 
variable `y`, the prior variance `priorVar` and the number of data points 
`nData`. Strictly speaking, `nData` is not a requirement, since it can be 
deduced by `size(X, 1)`. It is however more efficient to pass `nData` to the 
`data` dictionary, given that a typical MCMC simulation invokes the `Model` 
functions thousands of times.

As a concrete example, `test/swiss.txt` contains the Swiss banknotes dataset. 
The dataset holds the measurements of 4 covariates on 200 Swiss banknotes, 
of which 100 genuine and 100 counterfeit, representing the length of the bill, 
the width of the left and the right edge, and the bottom margin width. 
Therefore, the design matrix `X` has 200 rows and 4 columns, and the 4 
regression coefficients constitute the model parameters to be estimated. The 
binary response variable `y` is the type of banknote, 0 being genuine and 1 
counterfeit. To create the `data` dictionary for the Swiss banknotes, change 
into the `test` directory and run the `test/swiss.jl` script:

    # Create design matrix X and response variable y from swiss data array
    swiss = readdlm("swiss.txt", ' ');
    covariates = swiss[:, 1:end-1];
    nData, nPars = size(covariates);

    covariates = (bsxfun(-, covariates, mean(covariates, 1))
    ./repmat(std(covariates, 1), nData, 1));

    polynomialOrder = 1;
    X = zeros(nData, nPars*polynomialOrder);
    for i = 1:polynomialOrder
      X[:, ((i-1)*nPars+1):i*nPars] = covariates.^i;
    end

    y = swiss[:, end];
    
    # Create data dictionary
    data = {"X"=>X, "y"=>y, "priorVar"=>100., "nData"=>nData};

#### Defining the functions of the `Model`

Having defined the `data` dictionary as above, `src/logitNormalPrior.jl` 
provides the functions of the Bayesian logit model:

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

There are two rules to bare in mind when defining the functions of the model. 
The first rule is to use the reserved names `logPrior`, `logLikelihood`, 
`gradLogPosterior`, `tensor`, `derivTensor` and `randPrior` in the relevant 
function definitions. The second rule is to adhere to the signature

    function myFunction(pars::Vector{Float64}, nPars::Int, data::Union(Array{Any}, Dict{Any, Any}))

for the deterministic functions of the model, and to the signature

    function myFunction(nPars::Int, data::Union(Array{Any}, Dict{Any, Any}))

for the stochastic function `randPrior`.

The second argument `nPars` to the model's functions seems redundant, since it 
can be derived from the first argument `pars` via the `size(pars, 1)` command. 
Nevertheless, it is more efficient practice to pass `nPars` as an argument 
given the number of times the functions are invoked in a single MCMC run. 
Besides, once the functions are passed to `Model`, then they are invoked with 
fewer arguments as members of the instantiated `Model`, so the interface remains 
simple.

#### Creating an instance of `Model`

Once the `data` and the functions of `Model` are defined, it is straightforward 
to instantiate `Model` with a single command. For example, 
`test/logitNormalPriorSwissRmhmc.jl` demosntrates how to create an instance of 
`Model` so as to run RMHMC:

    model = Model(nPars, data, logPrior, logLikelihood, gradLogPosterior,
      tensor, derivTensor, randPrior);

As shown in `test/logitNormalPriorSwissMmala.jl`, the same `Model` 
instantiation is used for MMALA. In fact, the same command invocation suffices 
to create an instance which can be then utilized by any of the 6 MCMC 
algorithms of the GeometricMCMC package.

In order to make `Model` more user friendly, it is possible to shorten the 
`Model` invocation in some cases. For example, the partial derivatives of the 
metric tensor are not needed when running SMMALA. This is why `derivTensor` is 
omitted in `test/logitNormalPriorSwissSmmala.jl` in the `model` definition:

    model = Model(nPars, data, logPrior, logLikelihood, gradLogPosterior, tensor, randPrior);

As a second example, the `tensor` function is not needed when running 
Metropolis-Hastings, MALA or HMC. For this reason, both `tensor` and 
`derivTensor` can be left out at `Model` instantiation. So, the `model` 
definition in `test/logitNormalPriorSwissMh.jl`, 
`test/logitNormalPriorSwissMala.jl` and`test/logitNormalPriorSwissHmc.jl`
takes the more succinct form

    model = Model(nPars, data, logPrior, logLikelihood, gradLogPosterior, randPrior);

Apparently, the corresponding function definitions `tensor` and `derivTensor` 
are not required if there is no intention to run RMHMC or MMALA.
      
### The MCMC option types

## Future features

The package is extended in order to allow usage of the MCMC routines with ODE 
models.
