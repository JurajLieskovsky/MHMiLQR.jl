# MHMiLQR.jl

MHMiLQR.jl is a Julia package for unconstrained trajectory optimization. It implements regularization using minimal Hessian modification (MHM) on top of the iterative linear-quadratic regulator (iLQR). This form of regularization makes the algorithm particularly adept at solving highly non-convex problems.

The implementation also efficiently handles states that lie on a manifold, such as the SO(3) group in the case of quaternions, by projecting partial derivatives onto the manifold.

## Solved problem

In particular, the algorithm solves the unconstrained trajectory optimization problem
```math
\begin{aligned}
\min_{x_{0:N}, u_{0:N-1}} &\enspace \Phi(x_N, N) + \sum_{k=0}^{N-1} l(x_k,u_k,k) \\
\text{s.t.} &\enspace x_{k+1} = f(x_k,u_k,k), \quad \forall k \in \{0, \dots, N-1\} \\
			&\enspace x_0 = \tilde{x}_0,
\end{aligned}
```
where $x_k$ and $u_k$ are the state and input of the system, $\Phi$ is the final cost; $l$ is the running cost; $f$ are the discrete-time dynamics of the system; $\tilde{x}_0$ is the initial state; and $N \in \mathbb{N}$ is the length of the horizon.
