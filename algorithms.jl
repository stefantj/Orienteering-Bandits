# Contains learners.

#using NIPS_sim.BanditProblem, NIPS_sim.solve_OP, NIPS_sim.solve_submod_OP

# CombLinTS
# Proposed by Kveton et al.
function CombLinTS(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    reward = zeros(T)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    for t = 1:T
	print(".")
        # Plan action:
        path = solve_OP(GPR.sample_n(posterior, problem.locations'), problem.distances, problem.budget, problem.n_start, problem.n_stop)
        # Sample path
        for pt in path
            y = problem.weights[pt] + sqrt(problem.prior.noise)*randn();
            GPR.update(posterior, problem.locations[pt,:]', y);
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


        path = solve_OP(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop)
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
    C = 1

    for t = 1:T
        beta_t = sqrt(2*log(t^2*N*(pi^2)/0.6))
        path = []

        if(information > C)
            path = solve_submod_OP(problem, beta_t, posterior)
        else
            println("Exploring for more data $information $t")
            # This is a bad idea, but stop-gap.
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
    for t = 1:T
	print("_")
        budget_left = deepcopy(problem.budget)
        path_taken = [problem.n_start];
        distances = deepcopy(problem.distances)
        k = 0;
        while(path_taken[end] != problem.n_stop)
            # Update with sample at current location
            GPR.update(posterior, problem.locations[path_taken[end],:]', problem.weights[path_taken[end]] + sqrt(problem.prior.noise)*randn())
            # Plan action:
            k+=1;
            ucb = zeros(N);
            beta_tk = sqrt(2*log( (t)^2 * N * (pi^2) / 0.6))

            for i=1:N
                mean,var = GPR.predict(posterior, problem.locations[i,:]')
                ucb[i] = mean;
                if(!isnan(var))
                    ucb[i] += beta_tk*sqrt(var)
                else
                    ucb[i] += sqrt(2);
                end
            end
# TODO: Fix this dumbness
	    tmp_problem = BanditProblem(problem.G, problem.locations, problem.distances, problem.prior, problem.weights, budget_left, path_taken[end], problem.n_stop)
            path = solve_submod_OP(tmp_problem, beta_tk, posterior)
            path_taken = [path_taken; path[2]]
            budget_left -= problem.distances[path_taken[end-1], path_taken[end]]
            # Mark node visited to avoid cycles
            if(length(path_taken) > 1)
                distances[:,path_taken[end-1]] = problem.budget
                distances[path_taken[end-1],:] = problem.budget
            end
        end
        reward[t] = sum(problem.weights[path_taken]);
    end

    return reward
end


