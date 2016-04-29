# Script for running the simulation.

# has functions for plotting
include("plot_utils.jl")

# Has the main simulation code.
include("CombinatorialBandits.jl")

# Load simulator on all processes
@everywhere using CombinatorialBandits

# Main routine for running the simulation:
function Bayesian_regret_simulation(NUM_ITERS, T_HORIZON)
    println("Running Bayesian regret simulator, averaging over $NUM_ITERS bandit problems with time horizon $T_HORIZON");
    
    # Initialize data capture
    problem_size = 10;
    Average_Regret = zeros(4,T_HORIZON)
    Squared_Regret = zeros(4,T_HORIZON)

    Average_Regret, Squared_Regret = CombinatorialBandits.Bayesian_regret_lattice(problem_size, NUM_ITERS, T_HORIZON)

    initialize_plots()
    pretty_plot(1, vec(Average_Regret[1,:]), vec(Squared_Regret[1,:]))
    pretty_plot(2, vec(Average_Regret[2,:]), vec(Squared_Regret[2,:]))
    pretty_plot(3, vec(Average_Regret[3,:]), vec(Squared_Regret[3,:]))
    pretty_plot(4, vec(Average_Regret[4,:]), vec(Squared_Regret[4,:]))

    ax=gca();
    fig=gcf();
    ax[:patch][:set_visible](false)
    fig[:canvas][:draw]()
end

# Simulate regret for problem sizes
# TODO: Make the graph structure random, current results depend way too much on the structure.
function Problem_size_simulation(NUM_ITERS, T_HORIZON, kvals)
    println("Running problem size simulator, averaging over $NUM_ITERS, with time horizon $T_HORIZON. $kvals will be taken");
    
    R_K = zeros(4,length(kvals));
    V_K = zeros(4,length(kvals));
    k_index = 0;

    for K in kvals
        println("=====================");
        println("Problem size: 3 x $K")
        println("=====================");
        Average_Regret, Squared_Regret = CombinatorialBandits.Bayesian_regret_trench(K, NUM_ITERS, T_HORIZON)
        k_index +=1
        for i = 1:4
            # This is the regret at time T
            R_K[i, k_index] = sum(vec(Average_Regret[i,:]'))
        end
    end

    println(R_K);

    initialize_plots();
    figure(15); clf();
    plot(kvals, R_K');
    legend(["CombLinTS", "CombLinUCB", "CombGPUCB", "SeqCombGPUCB"])

    return R_K
end


