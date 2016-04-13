# Script for running the simulation.

# has functions for plotting
include("plot_utils.jl")

# Has the main simulation code.
include("nips_sim.jl")

# Load simulator on all processes
@everywhere using NIPS_sim



# Main routine for running the simulation:
function Bayesian_regret_simulation(NUM_ITERS, T_HORIZON)
    println("Running Bayesian regret simulator, averaging over $NUM_ITERS bandit problems with time horizon $T_HORIZON");
    
    # Initialize data capture
    problem_size = 4;
    Average_Regret = zeros(4,T_HORIZON)
    Squared_Regret = zeros(4,T_HORIZON)

    Average_Regret, Squared_Regret = NIPS_sim.Bayesian_regret(problem_size, NUM_ITERS, T_HORIZON)

    # Print just in case plotting goes wrong...
    println(Average_Regret)
    println(Squared_Regret)

    initialize_plots()
    pretty_plot(1, vec(Average_Regret[1,:]), vec(Squared_Regret[1,:]))
    pretty_plot(2, vec(Average_Regret[2,:]), vec(Squared_Regret[2,:]))
    pretty_plot(3, vec(Average_Regret[3,:]), vec(Squared_Regret[3,:]))
 #       pretty_plot(4, vec(Average_Regret[4,:]), vec(Squared_Regret[4,:]))

    ax=gca();
    fig=gcf();
    ax[:patch][:set_visible](false)
    fig[:canvas][:draw]()

    return Average_Regret, Squared_Regret;
end


# Simulate regret for problem sizes
# TODO: Make the graph structure random, current results depend way too much on the structure.
function Problem_size_simulation(NUM_ITERS, T_HORIZON, K_steps)
    println("Running problem size simulator, averaging over $NUM_ITERS, with time horizon $T_HORIZON. $K_steps will be taken");

    R_K = zeros(4,K_steps);
    V_K = zeros(4,K_steps);
    k_index = 0;

    for K = 4:(4+K_steps-1)
        println("=====================");
        println("Problem size: $K x $K")
        println("=====================");
        Average_Regret, Squared_Regret = NIPS_sim.Bayesian_regret(K, NUM_ITERS, T_HORIZON)
        println("A = ", size(Average_Regret), " S = ", size(Squared_Regret))
        k_index +=1
        for i = 1:4
            R_K[i, k_index] = sum(vec(Average_Regret[i,:]'))
        end
    end
    return R_K, V_K
end


