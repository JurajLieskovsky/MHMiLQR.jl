# MHMiLQR.jl - Minimal Hessian Modification iLQR
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://jurajlieskovsky.github.io/MHMiLQR.jl/dev/)

MHMiLQR.jl is a Julia package for unconstrained trajectory optimization. It implements regularization using minimal Hessian modification (MHM) on top of the iterative linear-quadratic regulator (iLQR), as proposed in [[1]](#1). This form of regularization makes the algorithm particularly adept at solving highly non-convex problems.

The implementation also efficiently handles states that lie on a manifold, such as the SO(3) group in the case of quaternions, by projecting partial derivatives onto the manifold.

## Metadata

```
Title: MHMiLQR.jl - Minimal Hessian Modification iLQR
ID: MHMiLQR25
Version: 0.1.0
Project: Robotics and Advanced Industrial Production
Project No.: CZ.02.01.01/00/22_008/0004590
Project RO: 1.1-Optimal control of interconnected time-delay systems
Date: 25.11.2025
Authors: Juraj Lieskovský
Keywords: trajectory optimization, optimal control, dynamic programming
```

---

<a id="1">[1]</a>
Lieskovský, J., Bušek, J., and Vyhlídal, T. (2025). iLQR Regularization using Minimal Hessian Modification. In: 2025 European Control Conference (ECC), preprint submitted Nov 14, 2025.
