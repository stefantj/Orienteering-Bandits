# Simulators.jl

# Runs a Bayesian regret simulation on the given problem.
function Bayesian_Regret(PROBLEM, NUM_ITERS, T_HORIZON)

    All_Regrets    = zeros(4, T_HORIZON, NUM_ITERS)
    Average_Regret = zeros(4, T_HORIZON)
    Squared_Regret = zeros(4, T_HORIZON)

    # Create unique file name for storing data:
    filename = "data/"*randstring()*".jld"

    println("Saving to $filename")
    p_size = 0;
    t_avg = 0;
    for iteration = 1:NUM_ITERS
        tic()
        println("\n==== ITER $iteration ====");
## Generate instance of the problem from the prior: ##
	problem_data = PROBLEM
        problem_data = resample_bandit_problem(PROBLEM)
#        problem_data = shuffle_bandit_problem(resample_bandit_problem(PROBLEM))
#        optimal_path = solve_OP(problem_data);
#        p_size += length(optimal_path)
#        println("Optimal path has ", length(optimal_path), " nodes. Avg = ",p_size/iteration)
        optimal_path = solve_dijkstra(problem_data)

## Run learning algorithms: ##

        println("Optval = ", sum(problem_data.weights[optimal_path]))
        # CombLinTS: From Kveton et al.
        s_comblints = @spawn CombLinTS(problem_data, T_HORIZON)

        # CombLinUCB: From Kveton et al.
#        s_comblinucb = @spawn CombLinUCB(problem_data, T_HORIZON)

        # CombGPUCB: Algorithm 1 from our paper
#        s_combgpucb = @spawn CombGPUCB(problem_data, T_HORIZON)

        # SeqCombGPUCB: Algorithm 2 from our paper
        s_seqcombgpucb = @spawn SeqCombGPUCB(problem_data, T_HORIZON)

## Record cumulative rewards: ##
        r_optimal = sum(problem_data.weights[optimal_path])
        r_comblints = fetch(s_comblints)
#        r_comblinucb = fetch(s_comblinucb)
#        r_combgpucb  = fetch(s_combgpucb)
        r_seqcombgpucb = fetch(s_seqcombgpucb)
#        r_seqcombgpucb = @time SeqCombGPUCB(problem_data, T_HORIZON)
#        r_comblints  = NaN*r_seqcombgpucb
        r_comblinucb = NaN*r_seqcombgpucb
        r_combgpucb = NaN*r_seqcombgpucb

        All_Regrets[:, :,iteration] = [vec(r_optimal - r_comblints)'
                                       vec(r_optimal - r_comblinucb)'
                                       vec(r_optimal - r_combgpucb)'
                                       vec(r_optimal - r_seqcombgpucb)']

        println("Regrets: TS: ", sum(All_Regrets[1,:,iteration]), ", Seq: ", sum(All_Regrets[4,:,iteration]))
        println("Rewards: TS: ", sum(r_comblints), ", Seq: ", sum(r_seqcombgpucb))


## Record regret: ##
        Average_Regret[1,:] += ( (r_optimal - r_comblints)./NUM_ITERS)'
        Squared_Regret[1,:] += (((cumsum(r_optimal*ones(T_HORIZON)) - cumsum(r_comblints)).^2)./NUM_ITERS)'
        Average_Regret[2,:] += ( (r_optimal - r_comblinucb)./NUM_ITERS)'
        Squared_Regret[2,:] += (((cumsum(r_optimal*ones(T_HORIZON)) - cumsum(r_comblinucb)).^2)./NUM_ITERS)'
        Average_Regret[3,:] += ( (r_optimal - r_combgpucb)./NUM_ITERS)'
        Squared_Regret[3,:] += (((cumsum(r_optimal*ones(T_HORIZON)) - cumsum(r_combgpucb)).^2)./NUM_ITERS)'
        Average_Regret[4,:] += ( (r_optimal - r_seqcombgpucb)./NUM_ITERS)'
        Squared_Regret[4,:] += (((cumsum(r_optimal*ones(T_HORIZON)) - cumsum(r_seqcombgpucb)).^2)./NUM_ITERS)'

## Save file: ##
        save(filename, "problem_size", length(PROBLEM.weights), "problem", PROBLEM,"T", T_HORIZON, "iters", iteration, "Average_Regret", Average_Regret, "Squared_Regret", Squared_Regret, "All_Regrets", All_Regrets)
        t_this = toq()
        t_avg += t_this
        hrs_this = int(floor(t_this/3600))
        mins_this = int(t_this/60 - 60*hrs_this)

        t_left = (NUM_ITERS-iteration)*t_avg/iteration
        hrs = int(floor(t_left/3600))
        mins =int(t_left/60 - 60*hrs)
        println("Simulation took $hrs_this and $mins_this. Estimated remaining time is $hrs hours and $mins minutes.")
    end
    return Average_Regret, Squared_Regret
end

function Bayesian_regret_lattice(PROBLEM_SIZE, NUM_ITERS, T_HORIZON)
    return Bayesian_Regret(initialize_lattice_problem(PROBLEM_SIZE), NUM_ITERS, T_HORIZON)
end

function Bayesian_regret_trench(PROBLEM_WIDTH, PROBLEM_SIZE, NUM_ITERS, T_HORIZON)
    return Bayesian_Regret(initialize_trench_problem(PROBLEM_WIDTH,PROBLEM_SIZE), NUM_ITERS, T_HORIZON)
end

function Bayesian_regret_wind(FILENAME, NUM_ITERS, T_HORIZON)
    return Bayesian_Regret(initialize_wind_problem(FILENAME), NUM_ITERS, T_HORIZON)
end

function Bayesian_regret_chain(PROBLEM_WIDTH, PROBLEM_SIZE, NUM_ITERS,T_HORIZON)
    return Bayesian_Regret(initialize_chained_lattice_problem(PROBLEM_WIDTH,PROBLEM_SIZE), NUM_ITERS, T_HORIZON)
end


function test_graph_solutions()
    
    for size = 3:30
        # Create instances of problem
        d0 = @timed initialize_lattice_problem(size)

        # Re-sample problem
        d1 = @timed resample_bandit_problem(d0[1])

        problem = d1[1]
        # Initial solve
        d2 = @timed solve_OP_hotstart(problem.weights, problem.distances, problem.budget, problem.n_start, problem.n_stop, nothing, nothing)
        data = d2[1]
        println("Path 1 = ", data[1])
        r2 = sum(problem.weights[data[1]])
        # Hotstart solve
        d3 = @timed solve_OP_hotstart(problem.weights, problem.distances, problem.budget, problem.n_start, problem.n_stop, nothing, nothing)
        data = d3[1]
        println("Path 2 = ", data[1])
        r3 = sum(problem.weights[data[1]])
        # Dijkstra solve
        d4 = @timed solve_dijkstra(problem.G, problem.weights, problem.n_start, problem.n_stop)
        println("Path 3 = ", d4[1])
        r4 = sum(problem.weights[d4[1]])

        println("Size $size: Initialization time ", d0[2], " Sampling time ", d1[2], " Initial OP time: ", d2[2], " Second hotstart time: ", d3[2], " Dijkstra's time: ", d4[2])
        println("Rewards: $r2, $r3, $r4")

    end

end

