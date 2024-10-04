# Getting started

## Installation
The package can be added to your project using the command
```
] add https://github.com/JurajLieskovsky/MHMiLQR.jl.git
```
in your REPL. For usage, we recommend taking a look at the examples.

## Running Examples

To run the examples we recommend cloning main repository
```
git clone git@github.com:JurajLieskovsky/MHMiLQR.jl.git MHMiLQR
```
 Navigating to the `MHMiLQR/examples` folder and starting Julia
```
cd MHMiLQR/examples
julia
```
In the REPL, activating and instantiating the environment
```
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
Running one of the examples
```
include("cartpole/swing_up.jl")
```
