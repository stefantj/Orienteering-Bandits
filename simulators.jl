# Simulators.jl

# Runs a Bayesian regret simulation on the given problem.
function Bayesian_Regret(PROBLEM, NUM_ITERS, T_HORIZON)

    Average_Regret = zeros(4, T_HORIZON)
    Squared_Regret = zeros(4, T_HORIZON)

    # Create unique file name for storing data:
    filename = "data/"*randstring()*".jld"

    for iteration = 1:NUM_ITERS
        println("\n==== ITER $iteration ====");
## Generate instance of the problem from the prior: ##
        problem_data = resample_bandit_problem(PROBLEM)
        optimal_path = solve_OP(problem_data);

## Run learning algorithms: ##

        # CombLinTS: From Kveton et al.
        s_comblints = @spawn CombLinTS(problem_data, T_HORIZON)

        # CombLinUCB: From Kveton et al.
        s_comblinucb = @spawn CombLinUCB(problem_data, T_HORIZON)

        # CombGPUCB: Algorithm 1 from our paper
        s_combgpucb = @spawn CombGPUCB(problem_data, T_HORIZON)

        # SeqCombGPUCB: Algorithm 2 from our paper
        s_seqcombgpucb = @spawn SeqCombGPUCB(problem_data, T_HORIZON)

## Record cumulative rewards: ##
        r_optimal = sum(problem_data.weights[optimal_path])
        r_comblints = fetch(s_comblints)
        r_comblinucb = fetch(s_comblinucb)
        r_combgpucb  = fetch(s_combgpucb)
        r_seqcombgpucb = fetch(s_seqcombgpucb)
## Record regret: ##
        Average_Regret[1,:] += ( (r_optimal - r_comblints)./NUM_ITERS)'
        Squared_Regret[1,:] += (((r_optimal - r_comblints).^2)./NUM_ITERS)'
        Average_Regret[2,:] += ( (r_optimal - r_comblinucb)./NUM_ITERS)'
        Squared_Regret[2,:] += (((r_optimal - r_comblinucb).^2)./NUM_ITERS)'
        Average_Regret[3,:] += ( (r_optimal - r_combgpucb)./NUM_ITERS)'
        Squared_Regret[3,:] += (((r_optimal - r_combgpucb).^2)./NUM_ITERS)'
        Average_Regret[4,:] += ( (r_optimal - r_seqcombgpucb)./NUM_ITERS)'
        Squared_Regret[4,:] += (((r_optimal - r_seqcombgpucb).^2)./NUM_ITERS)'

## Save file: ##
        save(filename, "problem_size", length(PROBLEM.weights), "problem", PROBLEM,"T", T_HORIZON, "iters", iteration, "Average_Regret", Average_Regret, "Squared_Regret", Squared_Regret)
    end
    return Average_Regret, Squared_Regret
end

function Bayesian_regret_lattice(PROBLEM_SIZE, NUM_ITERS, T_HORIZON)
    return Bayesian_Regret(initialize_lattice_problem(PROBLEM_SIZE), NUM_ITERS, T_HORIZON)
end

function Bayesian_regret_trench(PROBLEM_SIZE, NUM_ITERS, T_HORIZON)
    return Bayesian_Regret(initialize_trench_problem(PROBLEM_SIZE), NUM_ITERS, T_HORIZON)
end
