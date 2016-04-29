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
    nominal_path = solve_OP(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop)


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


    second_path = solve_OP(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop)

    if(sum(abs(nominal_path - second_path)) != 0)
        println("Lazy evaluation does not guarantee correct answer")
    end

    return nominal_path
end


