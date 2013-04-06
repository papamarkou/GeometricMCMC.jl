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
two input arguments. The first argument is an instance of the Model type, 
defined in the package, and is common across all 6 MCMC routines. The second 
argument is an instance of the algorithm's options type and is specific to the 
algorithm.

### The Model type

Coming soon.

## Future features

Coming soon.

### The Options types

The package is extended in order to allow usage of the MCMC routines with ODE 
models.
