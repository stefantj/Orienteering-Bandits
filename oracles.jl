# Oracles for solving various combinatorial problems.

# Used for solving initial problem
function solve_OP(problem::BanditProblem)
    return solve_OP(problem.weights, problem.distances, problem.budget, problem.n_start, problem.n_stop)
end

# If the problem can be solved using Dynamic programming, do it
function solve_dijkstra(problem::BanditProblem)
    if(problem.is_DAC)
       return solve_dijkstra(problem.G, problem.weights, problem.n_start, problem.n_stop) 
    else
        println("Not acyclic, returning empty path")
        return []
    end
end

# Used if the problem is DAC
# Takes in node weights, transforms into edge weights, and solves using dijkstra's
# Returns the longest weight path.
function solve_dijkstra(G::GenericGraph, node_weights, start_node, stop_node)
    num_edges = length(G.edges)
    edge_weights = zeros(num_edges)
    edge_index = 0
    for edge in G.edges
        edge_index+=1
        edge_weights[edge_index] = -node_weights[edge.target]
    end
    edge_weights -= minimum(edge_weights)

    # solve problem
    ssp = Graphs.dijkstra_shortest_paths(G, edge_weights, start_node)

    # Reconstruct path
    path = [stop_node]
    while(path[1] != start_node)
        path = [ssp.parents[path[1]], path]
    end
    return path
end

# Used for solving the modular orienteering problem. Casts as a MIP
# requires Gurobi
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
                path = [path; curr]
                curr = find(x_sol[curr,:]);
            end
            path = [path; curr]
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
                path = [path; curr]
                curr = find(x_sol[curr,:]);
            end
            path = [path; curr]
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


type pathObject
    path::Vector{Int64}
    budget_left::Float64
    reward_to_curr::Float64    
end

# Used for solving submodular OP. Uses Branch and bound.
function solve_submod_OP(problem::BanditProblem, beta, prior::GPR.GaussianProcessEstimate, x, u)

    N = length(problem.G.vertices)
    gp = deepcopy(prior)

    tic()
    # First, try lazily evaluating the path
    ucb = GPR.form_ucb(gp,problem.locations', (beta*beta))

    t0 = toq()
    tic()
    x_nominal = nothing
    u_nominal = nothing
    nominal_path = []
    if(problem.is_DAC)
        nominal_path = solve_dijkstra(problem.G, ucb, problem.n_start, problem.n_stop)
    else
        nominal_path, x_nominal, u_nominal= solve_OP_hotstart(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop, x, u)
    end
    t1=toq()
    #=
# We skip the re-check since it significantly increases run-time and 99.99% of the time is irrelevant.
    tic()
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
    t2 = toq()
    tic()

    second_path = []
    if(problem.is_DAC)
        second_path = solve_dijkstra(problem.G, ucb, problem.n_start, problem.n_stop)
    else
        second_path, x_nominal, u_nominal = solve_OP_hotstart(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop, x_nominal, u_nominal)
    end
    t3 =toq()

    println("$t0 $t1 $t2 $t3;")

    if(length(nominal_path) != length(second_path) || sum(abs(nominal_path - second_path)) != 0)
        println("Lazy evaluation does not guarantee correct answer: Trying branch and bound")
        n_t = problem.n_stop
        LB = sum(problem.weights[nominal_path])
        HHeap = Collections.PriorityQueue()
        HHeap[pathObject([problem.n_start], problem.budget, ucb[problem.n_start])] = 0.
        while(!isempty(HHeap))
            p = Collections.dequeue!(HHeap)
            curr = p.path
            budget_left = p.budget_left
            reward_to_curr = p.reward_to_curr
            if(curr[end]==n_t) # Reached the destination so compute exact reward
                reward = sum(problem.weights[curr])+beta*compute_information(problem, prior, curr)
                if(reward > LB)
                    #println("New best found ($reward)")
                    LB = reward
                    nominal_path = curr
                end

            else
                print(".")

                # Here we reduce the problem:
                nodes_visited = curr[1:end-1]
                skip=0;
                unvisited = []
                gp = deepcopy(prior)

                opt2_path = []
                for k = 1:length(problem.weights)
                    if(k in nodes_visited)
                        skip+=1
                        GPR.update(gp, problem.locations[k, :]', 0.)
                    else
                        unvisited = [unvisited; k]
                    end
                end

                # Compute UCB for unvisited sites
                ucb_r = zeros(length(unvisited))
                i = 0;
                for k in unvisited
                    i+=1
                    m,v = GPR.predict(gp, problem.locations[k,:]')
                    if(isnan(v))
                        v=1
                    end
                    ucb_r[i] = m + beta*sqrt(v)
                end
                
                # Compute upper bounds
                for edge in problem.G.finclist[curr[end]]
                    next = edge.target

                    if(next in unvisited)
                      # convert indices to reduced problem
                      n_s_r = find(unvisited.==curr[end])[1]
                      n_t_r = find(unvisited.==problem.n_stop)[1]

                      if(problem.is_DAC)
                          prb = subproblem(problem,unvisited)
                        newpath = solve_dijkstra(prb.G, ucb_r, n_s_r, n_t_r)
                      else
                        newpath = solve_OP(ucb_r, problem.distances[unvisited, unvisited], budget_left, n_s_r, n_t_r)
                      end
                      if(!isempty(newpath))
                        if(length(unvisited)==0)
                            #println("no unvisited nodes")
                        end
#                        println("UCBR = $ucb_r")
                        reward_to_next = reward_to_curr + ucb_r[find(unvisited.==next)][1]
                        U_next = reward_to_curr + sum(ucb_r[newpath])
                        if(U_next > LB)
#			println("*")
#                             println("Checking types:\n path: ", typeof(vec([curr; int(next)])))
#                             println("budget_left: ", typeof( budget_left-problem.distances[curr[end],next]))
#                             println("reward_to_next: $reward_to_next", typeof(reward_to_next))
#                             println("Constructing object:", pathObject(vec([curr; int(next)]), budget_left-problem.distances[curr[end],next], reward_to_next))
                             HHeap[pathObject(vec([curr; int(next)]), budget_left-problem.distances[curr[end],next], reward_to_next)] = -U_next

                        else 
                             #println("eliminated path $curr")
                        end
                      else
                        #println("$curr closed")
                      end
                  end
                end

            end
        end
    end
    =#
    return nominal_path, x_nominal, u_nominal
end


