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

## Installation



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

For ease of use, all the user-defined functions in the Model type share the 
same signature

    function myFunction(pars::Vector{Float64}, nPars::Int, data::Dict{Any, Any})

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

## An example: the Bayesian logit model with Normal prior

### The MCMC option types

Coming soon.

## Future features

The package is extended in order to allow usage of the MCMC routines with ODE 
models.
