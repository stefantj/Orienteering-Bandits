# Script for running the simulation.

# has functions for plotting
include("plot_utils.jl")

# Has the main simulation code.
include("CombinatorialBandits.jl")

# Load simulator on all processes
@everywhere using CombinatorialBandits

# Run a typical stacked lattice simulation
println("Running simulation with 10 iterations of 500 time epochs. The problem structure is a stacked lattice with 15 3x3 lattices.")
CombinatorialBandits.Bayesian_regret_chain(15,3,10,500)
plot_datafile("sim_data.jld")

