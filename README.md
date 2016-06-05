# Orienteering-Bandits
Simulation code for Stanford EE378A course project. DISCLAIMER: this code is a prototype and there are absolutely no guarantees on correctness (beyond "I _think_ it's correct")

The purpose of this code is to test algorithms for a new setting of multi-armed bandits, where sequences of K actions must satisfy combinatorial constraints but the learner can update its action sequence. In this document, an iteration is one run of the simulator, which consists of T_HORIZON epochs, each containing K time steps.  

# Running the code
## Installation
Sorry for not making this an julia repo, but the code is simple so installation should be too. This software was written in Julia 0.3, but should be compatible with 0.4 (minus some plotting issues). This code requires the packages Graphs, JuMP, JLD, and PyPlot (for plotting). These can be installed by the command `Pkg.install("<package name>")`.
## Quick-start
After installation is complete, navigate to the Orienteering-Bandits directory and start the Julia REPL. The simulator can be run on a 15x3x3 problem by calling `include("run_sim.jl")`, which will take a moment because PyPlot is slow. After the iterations finish, a figure will pop up with the Bayesian regret versus time for both CombLinTS and SeqGPUCB.

## Running your own simulation 
A simulation is called using the function `CombinatorialBandits.Bayesian_regret_chain(PROBLEM_HEIGHT, PROBLEM_SIZE, NUM_ITERS)`, where `PROBLEM_WIDTH` is the number of levels in the stacked lattice, `PROBLEM_SIZE` is the number of vertices on each side of a level, `NUM_ITERS` is the number of iterations you want the simulator to average over, and `T_HORIZON` is the number of time steps in each iteration.
## Interpreting the data
Data is saved to sim_data.jld file, in the Orienteering-Bandits folder and is stored at the end of every iteration. The function plot_datafile(FILENAME) will plot the data in figure 1.
## Apologies
The code has a lot of bloat that needs to be taken out. If I have time (and am not sleeping), I will clean it up.

# Code organization (inasmuch as it exists)
## GPR
The Gaussian process regression code consists of three files. The header, GPR.jl, covariance functions, covariance.jl, and prediction functions in predict.jl. The implementation is based on Kernel recursive least squares, with an additional feature that collapses redundant data (this reduces the computation from O(t^2) to O(n^2)).

## Combinatorial Bandits
### Simulator
The outer loop of the simulator is contained in simulators.jl. This code initializes the problem, generates data and passes the necessary information to the algorithms. After the algorithms finish their iteration, the simulator saves data.

### Algorithms
There are four algorithms implemented in algorithms.jl. The only two that are relevant are CombLinTS (developed by Kveton et al.) and SeqCombGPUCB, developed in the course of this project. The algorithms rely on an oracle function to compute the actions, and the GPR module to compute the posterior.
### Oracles
The stacked lattice problem can be solved with dynamic programming. Oracles.jl also contains a mixed integer programming implementation of the orienteering problem, which can be used for more general graph types (but requires Gurobi to be installed on the machine). Technically the oracle is asked to solve a submodular maximization problem, but we take the approach recommended by Desautels in their paper "Parallelizing Exploration-Exploitation Tradeoffs in Gaussian Process Bandit Optimization", which first evaluates the path and then checks whether it is still optimal in the submodular setting. This checking is easily done because the submodular portion (the variance) is a monotonically decreasing function and so we have an upper bound on the value of all other actions. In practice the lazy method is correct 99.99% of the time, and so we skip it to halve the run-time.
### Problems
Three types of probems are listed in problems.jl: lattice, chain, and trench. Lattice is a 2D grid of points, and the trench problem is simply an asymmetric lattice. The chain problem is the stacked lattice problem which we describe in the report. I can only vouch for the correctness of the chain problem type. 
