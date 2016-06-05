# Orienteering-Bandits
Simulation code for Stanford EE378A course project.DISCLAIMER: his code is a prototype and there are absolutely no guarantees on correctness (beyond "I _think_ it's correct")

The main idea is to test algorithms for a new setting of multi-armed bandits, where sequences of K actions must satisfy combinatorial constraints but the learner can update its action sequence. 

# Running the code
## Installation
Sorry for not making this an julia repo, but the code is simple so installation should be too. This package was written in Julia 0.3 (sorry) and requires the packages Graphs, JuMP, JLD, and PyPlot (for plotting). These can be installed by the command `Pkg.install("<package name>")` etc.
## Quick-start
After installation is complete, navigate to the Orienteering-Bandits directory and start the Julia REPL. Load the necessary files by the command `include("run_sim.jl")`, which will take a moment because PyPlot is slow. To run one of the simulations used in my project, use the commands `CombinatorialBandits.Bayesian_regret_chain(PROBLEM_HEIGHT, PROBLEM_SIZE, NUM_ITERS)`, where `PROBLEM_WIDTH` is the number of levels in the stacked lattice, `PROBLEM_SIZE` is the number of vertices on each side of a level, `NUM_ITERS` is the number of iterations you want the simulator to average over, and `T_HORIZON` is the number of time steps in each iteration.
## Interpreting the data
Data is saved in a .jld file, in the Orienteering-Bandits/data folder. When a simulation is run, it generates a random filename (which is outputted to the screen) and stores the data at the end of ever iteration. The script plot_all_regrets.jl has functions useful for visualizing the results: `include("plot_all_regrets.jl"); plot_range_datafile("<name of datafile>.jld",[1,4])`. A figure will open with the Thompson sampling result shown in red and the SeqUCB result in blue. 
## Apologies
The code has a lot of bloat that needs to be taken out. If I have time (and am not sleeping), I will clean it up.
