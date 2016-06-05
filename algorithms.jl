# Contains learners.

#using NIPS_sim.BanditProblem, NIPS_sim.solve_OP, NIPS_sim.solve_submod_OP

# CombLinTS
# Proposed by Kveton et al.
function CombLinTS(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    reward = zeros(T)
    dim = size(problem.locations,2)
    posterior = GPR.GaussianProcessEstimate(problem.prior, dim);
    x = nothing
    u = nothing
    for t = 1:T
        tic()
        sample_pts = GPR.sample_n(posterior, problem.locations');
        t0 = toq()
        tic()
        # Plan action:
        path=[]
        if(problem.is_DAC)
            path = solve_dijkstra(problem.G, sample_pts, problem.n_start, problem.n_stop)
        else
            path, x, u = solve_OP_hotstart(sample_pts, problem.distances, problem.budget, problem.n_start, problem.n_stop, x, u)
        end
        t1 = toq()
        tic()
        # Sample path
        for pt in path
            y = problem.weights[pt] + sqrt(problem.prior.noise)*randn();
            GPR.update(posterior, problem.locations[pt,:]', y);
        end
        reward[t] = sum(problem.weights[path])
        t2 = toq()
        if(mod(t-1,10)==0)
            r = reward[t]
            println("TS($r) $t")#t): TS = ", t0+t1+t2, "R = ", reward[t])#T1 = $t1, T2 = $t2")
        end
    end
    return reward
end

# CombLinUCB
# Runs the algorithm proposed by Kveton et al
# Oracle problem solves static problem based on
# Runs the bandit algorithm over a single instance of a problem. Returns the averaged regret over T trials.
function CombLinUCB(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    dim = size(problem.locations,2)
    posterior = GPR.GaussianProcessEstimate(problem.prior, dim);
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

        path = []
        if(problem.is_DAC)
            path = solve_dijkstra(problem.G, GPR.sample_n(posterior, problem.locations'), problem.n_start, problem.n_stop)
        else
            path, x, u = solve_OP_hotstart(GPR.sample_n(posterior, problem.locations'), problem.distances, problem.budget, problem.n_start, problem.n_stop, x, u)
        end
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
    dim = size(problem.locations,2)
    posterior = GPR.GaussianProcessEstimate(problem.prior, dim);
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
    dim = size(problem.locations,2)
    posterior = GPR.GaussianProcessEstimate(problem.prior, dim)
    reward = zeros(T)
    information = 0;
    C = .3;
    delta = 0.9

    for t = 1:T
        tic()
        path_taken = [problem.n_start];
        path_queue = []
        distances = 0
        budget_left = 0
        if(!problem.is_DAC)
            # IN DAC problem, this is irrelevant since 1) no loops and 2) all paths to goal have same cost
            distances = deepcopy(problem.distances)
            budget_left = deepcopy(problem.budget)
        end
        k = 0;
        t0 = toq()
        while(path_taken[end] != problem.n_stop)
            tic()
            # Update with sample at current location
            GPR.update(posterior, problem.locations[path_taken[end],:]', problem.weights[path_taken[end]] + sqrt(problem.prior.noise)*randn())
            v = GPR.predict_var(posterior, vec(problem.locations[path_taken[end],:]'));
            information += 0.5*log(1+ v/posterior.prior.noise)

            # Check if we need to re-plan
            if(information > C || isempty(path_queue))
                if(information > C)
                    information = 0
                end

                # Plan action:
                beta_tk = (e^C)*sqrt(2*log( (t^2) * N * (pi^2) / (6*delta)))
                ucb = GPR.form_ucb(posterior, problem.locations', beta_tk^2)

# TODO: Fix this dumbness
                x = nothing
                u = nothing
                unvisited = setdiff([1:N], path_taken[1:end-1])
                # This creates graph
                tmp_problem = subproblem(problem, unvisited)
                n_s_r = find(unvisited.==path_taken[end])[1]
                n_t_r = find(unvisited.==problem.n_stop)[1]
                if(!problem.is_DAC)
	                tmp_problem = BanditProblem(tmp_problem.G, tmp_problem.locations, tmp_problem.distances, tmp_problem.prior, tmp_problem.weights, budget_left,n_s_r, n_t_r,problem.is_DAC)
                else
                    tmp_problem = BanditProblem(tmp_problem.G, tmp_problem.locations,tmp_problem.distances,tmp_problem.prior,tmp_problem.weights,tmp_problem.budget, n_s_r, n_t_r, problem.is_DAC)
                end

                path, x, u = solve_submod_OP(tmp_problem, beta_tk, posterior, x, u)
                # convert path back into `normal' indexing.
                path_queue = unvisited[path[2:end]]
            end

            # actually take path
            path_taken = [path_taken; shift!(path_queue)]
            if(!problem.is_DAC)
                budget_left -= problem.distances[path_taken[end-1], path_taken[end]]
                # Mark node visited to avoid cycles
                if(length(path_taken) > 1)
                    distances[:,path_taken[end-1]] = problem.budget
                    distances[path_taken[end-1],:] = problem.budget
                end
            end
            tmp = toq()
            t0 += tmp
#            print(" $tmp ")
        end
        reward[t] = sum(problem.weights[path_taken]);
        r = reward[t]
        if(mod(t-1,10)==0)
            println("\nSeq($r) $t")#: T = $t0, R = ", reward[t], ", ",posterior.numcenters)
        end
    end

    return reward
end


