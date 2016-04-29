# Oracles for solving various combinatorial problems.

# Used for solving initial problem
function solve_OP(problem::BanditProblem)
    return solve_OP(problem.weights, problem.distances, problem.budget, problem.n_start, problem.n_stop)
end


# TODO: Enable hot starts for solving. Requires re-structuring the return output.

# Used for solving the modular orienteering problem. Casts as a MIP
function solve_OP(values, distances, B,  n_s, n_t)
    # Formulate problem for Gurobi:
    N = length(values);
    without_start = [1:n_s-1; n_s+1:N];
    without_stop = [1:n_t-1; n_t+1:N];
    without_both  = intersect(without_start, without_stop);

    model = Model(solver=Gurobi.GurobiSolver(OutputFlag=0));

    # Indicator variables
    @defVar(model, x[1:N,1:N], Bin) # NxN binary variables - x[i,j] == 1 means j is visited just after i
    @defVar(model, 2 <= u[without_start] <= N, Int) # binary variable related to subtours
    # sum reward of visited nodes:
    @setObjective(model, Max, sum{ sum{values[i]*x[i,j], j=1:N}, i=1:N})
    # limit one child per parent
    @addConstraint(model, sum{x[n_s,j], j=without_start} == 1)
    @addConstraint(model, sum{x[i,n_t], i=without_stop} == 1)
    # path constraints/no cycles
    @addConstraint(model, connectivity[k=without_both], sum{x[i,k], i=1:N} == sum{x[k,j], j=1:N})
    @addConstraint(model, once[k=1:N], sum{x[k,j], j=1:N} <= 1)
    @addConstraint(model, sum{ sum{distances[i,j]*x[i,j], j=1:N}, i=1:N } <= B)
    @addConstraint(model, nosubtour[i=without_start,j=without_start], u[i]-u[j]+1 <= (N-1)*(1-x[i,j]))
    if(n_s != n_t)
        @addConstraint(model, sum{x[n_t,i],i=1:N}==0)
    end

    path = [];
    status = solve(model)
    if status != :Optimal
        warn("Not solved to optimality: \n")
    else 
        path = [n_s]
        x_sol = getValue(x);
        curr = find(x_sol[n_s,:]);
        if(length(curr) > 0)
            while(curr[1] != n_t)
                path = [path, curr]
                curr = find(x_sol[curr,:]);
            end
            path = [path, curr]
        end
    end

    return path
end


# Uses and returns hotstart values
function solve_OP_hotstart(values, distances, B, n_s, n_t, x_init, u_init)
    # Formulate problem for Gurobi:
    N = length(values);
    without_start = [1:n_s-1; n_s+1:N];
    without_stop = [1:n_t-1; n_t+1:N];
    without_both  = intersect(without_start, without_stop);

    model = Model(solver=Gurobi.GurobiSolver(OutputFlag=0));

    # Indicator variables
    @defVar(model, x[1:N,1:N], Bin) # NxN binary variables - x[i,j] == 1 means j is visited just after i
    @defVar(model, 2 <= u[without_start] <= N, Int) # binary variable related to subtours

    # Hot start if data
    if(!is(x_init, nothing))
        setValue(x, x_init)
        for keytuple in keys(u_init)
            key = keytuple[1]
            setValue(u[key], u_init[key])
        end
    end

    # sum reward of visited nodes:
    @setObjective(model, Max, sum{ sum{values[i]*x[i,j], j=1:N}, i=1:N})
    # limit one child per parent
    @addConstraint(model, sum{x[n_s,j], j=without_start} == 1)
    @addConstraint(model, sum{x[i,n_t], i=without_stop} == 1)
    # path constraints/no cycles
    @addConstraint(model, connectivity[k=without_both], sum{x[i,k], i=1:N} == sum{x[k,j], j=1:N})
    @addConstraint(model, once[k=1:N], sum{x[k,j], j=1:N} <= 1)
    @addConstraint(model, sum{ sum{distances[i,j]*x[i,j], j=1:N}, i=1:N } <= B)
    @addConstraint(model, nosubtour[i=without_start,j=without_start], u[i]-u[j]+1 <= (N-1)*(1-x[i,j]))
    if(n_s != n_t)
        @addConstraint(model, sum{x[n_t,i],i=1:N}==0)
    end

    path = [];
    status = solve(model)
    if status != :Optimal
        warn("Not solved to optimality: \n")
        return []
    else 
        path = [n_s]
        x_sol = getValue(x)
        u_sol = getValue(u)
        curr = find(x_sol[n_s,:]);
        if(length(curr) > 0)
            while(curr[1] != n_t)
                path = [path, curr]
                curr = find(x_sol[curr,:]);
            end
            path = [path, curr]
        end

        return path, x_sol, u_sol
    end
end

function compute_information(problem::BanditProblem, prior::GPR.GaussianProcessEstimate, path)
    gp = deepcopy(prior)
    information = 0;
    for x in path
        m,v = GPR.predict(gp, problem.locations[x,:]')
        GPR.update(gp,problem.locations[x,:]', 0.)
        if(isnan(v))
            v = 1
        end
        information += sqrt(v)
    end
    return information
end


# Used for solving submodular OP. Uses Branch and bound.
function solve_submod_OP(problem::BanditProblem, beta, prior::GPR.GaussianProcessEstimate)
    N = length(problem.G.vertices)
    gp = deepcopy(prior)
    # First, try lazily evaluating the path
    ucb = zeros(N)
    for i = 1:N
        mean, var = GPR.predict(gp, problem.locations[i,:]')
        ucb[i] = mean;
        if(!isnan(var))
            ucb[i] += beta*sqrt(var)
        else
            ucb[i] += 1;
        end
    end


    # TODO: use hot starts.
    nominal_path, x_nominal, u_nominal= solve_OP_hotstart(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop, nothing, nothing)


    # Now update prior and check if still optimal:
    ucb = GPR.predict_mean(gp, problem.locations');

    for x in nominal_path
        GPR.update(gp,problem.locations[x,:]',0.)
        m,v = GPR.predict(gp, problem.locations[x, :]')
        if(isnan(v))
            println("Error in prediction function")
        else
            ucb[x] += beta*sqrt(v)
        end
    end

    second_path = solve_OP_hotstart(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop, x_nominal, u_nominal)

    if(sum(abs(nominal_path - second_path)) != 0)
        println("Lazy evaluation does not guarantee correct answer: Trying branch and bound")
        
        # This is where you should insert BnB

        LB = sum(problem.weights[nominal_path])
        HHeap = Collections.PriorityQueue{Vector{Int64}, Vector{Float64}}(Base.Order.LexicographicOrdering())
        HHeap[[n_s]] = [0., B, ucb[n_s]]
        while(!isempty(HHeap))
            val = Collections.peek(HHeap)[2]
            budget_left = val[2]
            reward_to_curr = val[3]
            curr = Collections.dequeue!(HHeap)
            if(curr[end]==n_t) # Reached the destination so compute exact reward
                reward = sum(problem.weights[curr])+beta*compute_information(problem, prior, curr)
                if(reward > LB)
                    println("New best found ($reward)")
                    LB = reward
                    nominal_path = curr
                end

            else
                print(".")

                # Here we reduce the problem:
                nodes_visited = curr[1:end-1]
                skip=0;
                re_index = []
                unvisited = []
                ucb = []
                gp = deepcopy(prior)

                for k = 1:length(problem.weights)
                    if(k in nodes_visited)
                        skip+=1
                        GPR.update(gp, problem.locations[k, :]', 0.)
                    else
                        unvisited = [unvisited k]
                        re_index = [re_index k+skip]
                    end
                end

                # Compute UCB for unvisited sites
                ucb = zeros(length(unvisited))
                i = 0;
                for k in unvisited
                    i+=1
                    m,v = GPR.predict(gp, problem.locations[k,:]')
                    if(isnan(v))
                        v=1
                    end
                    ucb[i] = m + beta*sqrt(v)
                end
                
                # Compute upper bounds
                for edge in problem.G.finclist[curr[end]]
                    next = edge.target

                    n_s = re_index[curr[end]][1]
                    n_t = re_index[problem.n_stop][1]
                    newpath = solve_OP(ucb, problem.distances[unvisited, unvisited], budget_left, n_s, n_t)

                    if(!isempty(newpath))
                        reward_to_next = reward_to_curr + ucb[find(unvisited.==next)]
                        U_next = reward_to_curr + sum(ucb[newpath])
                        if(U_next > LB)
                             HHeap[[curr,next]] = vec([-U_next, budget_left - problem.distances[curr[end],next], reward_to_next]) 
                        else 
                             println("eliminated path $curr")
                        end
                    else
                        println("$curr closed")
                    end
                end

            end
        end
    end
    return nominal_path
end


