GeometricMCMC.jl
================================

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

Usage
-------------------------

To invoke each of the MCMC algorithms of the package, it is required to provide 
two input arguments. The first argument is an instance of the Model type, while 
the seconde one is an instance of the algorithm's options type.

As an example, the logit model with Normal prior is available with the package.

More detailed documentation will soon be available.

Future features
-------------------------

The package is extended in order to allow usage of the MCMC routines with ODE 
models.
