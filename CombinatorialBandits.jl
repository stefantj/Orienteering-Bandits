# Module file for combinatorial bandits simulations

module CombinatorialBandits

using JLD
using JuMP
#using Gurobi
using Graphs

include("GPR.jl")

# Other module files
include("problems.jl")
include("oracles.jl")
include("algorithms.jl")
include("simulators.jl")

export BanditProblem

end
