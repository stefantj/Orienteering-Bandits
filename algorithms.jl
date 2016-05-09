# Contains learners.

#using NIPS_sim.BanditProblem, NIPS_sim.solve_OP, NIPS_sim.solve_submod_OP

# CombLinTS
# Proposed by Kveton et al.
function CombLinTS(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    reward = zeros(T)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    x = nothing
    u = nothing
    for t = 1:T
        tic()
        sample_pts = GPR.sample_n(posterior, problem.locations');
        t0 = toq()
        tic()
        # Plan action:
        path, x, u = solve_OP_hotstart(sample_pts, problem.distances, problem.budget, problem.n_start, problem.n_stop, x, u)
        t1 = toq()
        tic()
        # Sample path
        for pt in path
            y = problem.weights[pt] + sqrt(problem.prior.noise)*randn();
            GPR.update(posterior, problem.locations[pt,:]', y);
        end
        t2 = toq()
        if(mod(t,100)==0)
            println("TS($t): TS = $t0, T1 = $t1, T2 = $t2")
        end
        reward[t] = sum(problem.weights[path])
    end
    return reward
end

# CombLinUCB
# Runs the algorithm proposed by Kveton et al
# Oracle problem solves static problem based on
# Runs the bandit algorithm over a single instance of a problem. Returns the averaged regret over T trials.
function CombLinUCB(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    reward = zeros(T)
    x = nothing
    u = nothing
    for t = 1:T
        # Plan action:
        beta = sqrt(2*log(t^2*N*(pi^2)/0.6))
        ucb = zeros(N)
        for i = 1:N
            mean, var = GPR.predict(posterior, problem.locations[i,:]')
            ucb[i] = mean;
            if(!isnan(var))
                ucb[i] += beta*sqrt(var)
            else
                ucb[i] += 1;
            end
        end


        path, x, u = solve_OP_hotstart(GPR.sample_n(posterior, problem.locations'), problem.distances, problem.budget, problem.n_start, problem.n_stop, x, u)
        # Now sample the path
        for i = 1:length(path)
            pt = path[i];
            GPR.update(posterior, problem.locations[pt,:]', problem.weights[pt] + sqrt(problem.prior.noise)*randn())
        end
        reward[t] = sum(problem.weights[path]);
    end
    return reward 
end

# CombGPUCB (algorithm 1)
# Same as CombLinUCB except:
# 1) The oracle solves the submodular problem
# 2) The coefficients are expanded to account for delayed feedback
function CombGPUCB(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    reward = zeros(T)
    information = 0
    C = log(2*sqrt(N) + 1)^3; # This is the maximum information that can be accumulated.  
    x = nothing
    u = nothing

    for t = 1:T
        beta_t = e*sqrt(2*log(t^2*N*(pi^2)/0.6))
        path = []

        if(information > C)
            path, x, u = solve_submod_OP(problem, beta_t, posterior, x, u)
        else
            # This is a bad idea, but stop-gap.
            println("C = $C, information = $information")
            path = solve_OP(zeros(length(problem.weights)), problem.distances, problem.budget, problem.n_start, problem.n_stop)
        end

        # Now sample the path
        for pt in path
            m,v = GPR.predict(posterior, problem.locations[pt,:]')
            GPR.update(posterior, problem.locations[pt,:]', problem.weights[pt] + sqrt(problem.prior.noise)*randn())
            if(isnan(v))
                v = 1
            end
            information += 0.5*log(1+ v/posterior.prior.noise)
        end
        reward[t] = sum(problem.weights[path]);
    end
    return reward 
end





# SeqCombGPUCB (algorithm 2)
# Runs the sequential combinatorial problem (with replanning)
function SeqCombGPUCB(problem::BanditProblem, T::Int64)
    # Extract problem data:
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2)
    reward = zeros(T)
    information = 0;
    C = 1;

    for t = 1:T
        tic()
        budget_left = deepcopy(problem.budget)
        path_taken = [problem.n_start];
        path_queue = []
        distances = deepcopy(problem.distances)
        k = 0;
        while(path_taken[end] != problem.n_stop)
            # Update with sample at current location
            GPR.update(posterior, problem.locations[path_taken[end],:]', problem.weights[path_taken[end]] + sqrt(problem.prior.noise)*randn())
            m,v = GPR.predict(posterior, problem.locations[path_taken[end],:]');
            if(isnan(v))
                v = 1
            end
            information += 0.5*log(1+ v/posterior.prior.noise)

            # Check if we need to re-plan
            if(information > C || isempty(path_queue))
                if(information > C)
                    information = 0
                end

                # Plan action:
                ucb = zeros(N);
                beta_tk = (e^C)*sqrt(2*log( (t^2) * N * (pi^2) / 0.6))

                new_C = 0;
                for i=1:N
                    mean,var = GPR.predict(posterior, problem.locations[i,:]')
                    ucb[i] = mean;
                    if(!isnan(var))
                        if(var > new_C)
                            new_C = var
                        end
                        ucb[i] += beta_tk*sqrt(var)
                    else
                        ucb[i] += 1
                    end
                end
                # Adjust C to its smallest value:
#                C = 0.5*log(1+new_C/posterior.prior.noise)

# TODO: Fix this dumbness
                x = nothing
                u = nothing
	            tmp_problem = BanditProblem(problem.G, problem.locations, problem.distances, problem.prior, problem.weights, budget_left, path_taken[end], problem.n_stop)
                path, x, u = solve_submod_OP(tmp_problem, beta_tk, posterior, x, u)
                path_queue = path[2:end]
            end

            # actually take path
            path_taken = [path_taken; shift!(path_queue)]
            budget_left -= problem.distances[path_taken[end-1], path_taken[end]]
            # Mark node visited to avoid cycles
            if(length(path_taken) > 1)
                distances[:,path_taken[end-1]] = problem.budget
                distances[path_taken[end-1],:] = problem.budget
            end
        end
        reward[t] = sum(problem.weights[path_taken]);
        t0=toq();
        if(mod(t,100)==0)
            println("\nSeq($t): T = $t0")
        end
    end

    return reward
end


