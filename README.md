# MHMiLQR.jl - Minimal Hessian Modification iLQR
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://jurajlieskovsky.github.io/MHMiLQR.jl/dev/)

MHMiLQR.jl is a Julia package for unconstrained trajectory optimization. It implements regularization using minimal Hessian modification (MHM) on top of the iterative linear-quadratic regulator (iLQR). This form of regularization makes the algorithm particularly adept at solving highly non-convex problems.

The implementation also efficiently handles states that lie on a manifold, such as the SO(3) group in the case of quaternions, by projecting partial derivatives onto the manifold.

