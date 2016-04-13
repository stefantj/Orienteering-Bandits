# Simulator for the NIPS paper
#

module NIPS_sim

# Used for estimation
include("../Libraries/GPR/GPR.jl")
using GPR

# Used for solving the planning rpoblem
using JuMP
using Gurobi
using Graphs

# Could make immutable if we split out optimal_path and weights into an `instance' variable.
# immutables are supposed to be good for speed
type BanditProblem
    G::GenericGraph                 # Node topology
    locations::Array{Float64,2}     # Node locations (in covariance space)
    distances::Array{Float64,2}     # Pairwise distances
    prior::GPR.GaussianProcess      # Prior distribution
    weights::Vector{Float64}        # Function weights
    budget::Float64                 # Travel budget
    n_start::Int64                  # Start node
    n_stop::Int64                   # Stop node
    optimal_path::Vector{Int64}     # Optimal path
end

function initialize_lattice_problem(pts_dim)

    num_pts = int(pts_dim^2)
    G = simple_graph(num_pts);
    edge_index = 0;
    node_index = 0;
    locs = linspace(0,1,pts_dim);
    locations = zeros(num_pts,2);

    for i = 1:pts_dim
        for j = 1:pts_dim
            node_index +=1
            locations[node_index,:] = [locs[i] locs[j]];
            if(j < pts_dim)
                edge_index+=1;
                add_edge!(G, Edge(edge_index, node_index, node_index+1))
            end
            if(i < pts_dim)
                edge_index+=1;
                add_edge!(G, Edge(edge_index, node_index, node_index+pts_dim))
            end
        end
    end

    # Budget doesn't really matter for a lattice problem.
    B = 2.19999;
    distances = 2*B*ones(num_pts,num_pts);
    for E in G.edges
        distances[E.source, E.target] = norm(locations[E.source,:] - locations[E.target,:]);
        if(!G.is_directed)
            distances[E.target,E.source] = distances[E.source, E.target]
        end
    end

    # Traverse diagonally
    n_s = 1;
    n_t = num_pts;
    # Initialize prior
    prior = GPR.GaussianProcess( 0.1, GPR.SquaredExponential(.44))

    weights = zeros(num_pts);
    optimal_path = zeros(2);

    return BanditProblem(G, locations, distances, prior, weights, B, n_s, n_t, optimal_path)
end
    
function solve_OP(problem::BanditProblem)
    return solve_OP(problem.weights, problem.distances, problem.budget, problem.n_start, problem.n_stop)
end

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

# Runs the bandit algorithm over a single instance of a problem. Returns the averaged regret over T trials.
function OP_Bandit(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    regret = ones(T)*sum(problem.weights[problem.optimal_path]);
    for t = 1:T
        # Plan action:
        means = GPR.predict_mean(posterior, problem.locations');
        path = solve_OP(means, problem.distances, problem.budget, problem.n_start, problem.n_stop)
        # Now sample the path
        for i = 1:length(path)
            pt = path[i];
            GPR.update(posterior, problem.locations[pt,:]', problem.weights[pt] + sqrt(problem.prior.noise)*randn())
        end
        regret[t] -= sum(problem.weights[path]);
    end
    return regret;
end

# Runs the bandit algorithm over a single instance of a problem. Returns the averaged regret over T trials.
# Uses UCB versus mean to plan. This is so far the best we've done
function OP3_Bandit(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    regret = ones(T)*sum(problem.weights[problem.optimal_path])
    for t = 1:T
        # Plan action:
        ucb = zeros(N);
        for i = 1:N
            mean,var = GPR.predict(posterior, problem.locations[i,:]');
            ucb[i] = mean
            if(!isnan(var))
                ucb[i] += sqrt(var);
            end
        end
        path = solve_OP(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop)
        # Now sample the path
        for pt in path
            GPR.update(posterior, problem.locations[pt,:]', problem.weights[pt] + sqrt(problem.prior.noise)*randn())
        end
        regret[t] -= sum(problem.weights[path]);
    end
    return regret;
end

# Runs the bandit algorithm over a single instance of a problem. Returns the averaged regret over T trials.
function OP2_Bandit(problem::BanditProblem, T::Int64)
    # Extract problem data:
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    regret = ones(T) * sum(problem.weights[problem.optimal_path])
    for t = 1:T
        budget_left = problem.budget
        path_taken = [problem.n_start];
        while(path_taken[end] != problem.n_stop)
            # Update with sample at current location
            GPR.update(posterior, problem.locations[path_taken[end],:]', problem.weights[path_taken[end]] + sqrt(problem.prior.noise)*randn())
            # Plan action:
            means = GPR.predict_mean(posterior, problem.locations');
            path = solve_OP(means, problem.distances, budget_left, path_taken[end], problem.n_stop)
            # In case of failure, see if we can finish the path manually and break.
            if(length(path) == 0) 
                if(path_taken[end] != problem.n_stop)
                    if(problem.distances[path_taken[end], problem.n_stop] <= budget_left)
                        path_taken = [path_taken, problem.n_stop]
                    end
                end
                break
            end
            path_taken = [path_taken, path[2]]
            budget_left -= problem.distances[path_taken[end-1], path_taken[end]]
        end
        # compute regret:
        regret[t] -= sum(problem.weights[path_taken]);
    end

    return regret;
end

function PS_Bandit(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    regret = ones(T)*sum(problem.weights[problem.optimal_path])
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    for t = 1:T
        # Plan action:
        path = solve_OP(GPR.sample_n(posterior, problem.locations'), problem.distances, problem.budget, problem.n_start, problem.n_stop)
        # Sample path
        for pt in path
            y = problem.weights[pt] + sqrt(problem.prior.noise)*randn();
            GPR.update(posterior, problem.locations[pt,:]', y);
        end
        # Compute regret
        regret[t] -= sum(problem.weights[path])
    end
    return regret;
end

function PS2_Bandit(problem::BanditProblem, T::Int64)
    println("\nResampling posterior bandit going to work...")
    # Extract problem data:
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    regret = ones(T)*sum(problem.weights[problem.optimal_path]);

    for t = 1:T
        print("-");
    end
    println();
    for t = 1:T
        print("=");
        budget_left = problem.budget
        path_taken = [problem.n_start];
        while(path_taken[end] != problem.n_stop)

            # Update with sample at current location
            GPR.update(posterior, problem.locations[path_taken[end],:]', problem.weights[path_taken[end]] + sqrt(problem.prior.noise)*randn())
            # Plan action:
            means = real(GPR.sample_n(posterior, problem.locations'))
            path = solve_OP(means, problem.distances, budget_left, path_taken[end], problem.n_stop)
            if(length(path) ==0 )
                break
            end
            path_taken = [path_taken, path[2]]
            budget_left -= problem.distances[path_taken[end-1], path_taken[end]]
        end
        
        # compute regret:

        regret[t] -=  sum(problem.weights[path_taken])
        for i = 1:length(path_taken)
            pt = path_taken[i];
            GPR.update(posterior, problem.locations[pt,:]', problem.weights[pt] + sqrt(problem.prior.noise)*randn())
        end
        if(regret[t] < 0)
            println("Better path found!! ");
        end
    end

    return regret
end


# Test:
#  OP hueristic with updates
#  PS with no updates
#  PS with updates <- expected to fail?
function Bayesian_regret(PROBLEM_SIZE, NUM_ITERS, T_HORIZON)

    problem_data = initialize_lattice_problem(PROBLEM_SIZE)
    Average_Regret = zeros(4, T_HORIZON)
    Squared_Regret = zeros(4, T_HORIZON)

    prior = GPR.GaussianProcessEstimate(problem_data.prior,2);
    for iteration = 1:NUM_ITERS
        println("==== ITER $iteration ====");
        # Sample function from the prior:
        # Consider having this be a separate vector, then problem_data can be immutable (possible speed-up?)
        problem_data.weights = vec(GPR.sample_n(prior, problem_data.locations'))
        problem_data.optimal_path = solve_OP(problem_data.weights, problem_data.distances, problem_data.budget, problem_data.n_start, problem_data.n_stop);

    # TODO: Implement each learner on a separate process.
        # Orienteering problem w/o replanning:
        s_OP = @spawn OP_Bandit(problem_data, T_HORIZON);

        # Orienteering problem w/replanning:
        #@time r_op2_learner = OP2_Bandit(problem_data, T_HORIZON);

        # Orienteering problem w/UCB:
        s_OP3 = @spawn OP3_Bandit(problem_data, T_HORIZON);

        # Posterior sampling w/o replanning:
        s_PS = @spawn PS_Bandit(problem_data, T_HORIZON);

        # Posterior sampling w/replanning:
#        r_ps2_learner = @spawn PS2_Bandit(problem_data,T_HORIZON);

        # Record results:
        r_op_learner = fetch(s_OP)
        r_op3_learner = fetch(s_OP3)
        r_ps_learner = fetch(s_PS)

        Average_Regret[1,:] += (r_op_learner./NUM_ITERS)'
        Squared_Regret[1,:] += ((r_op_learner.^2)./NUM_ITERS)'
        #Average_Regret[2,:] += (r_op2_learner./NUM_ITERS)'
        #Squared_Regret[2,:] += ((r_op2_learner.^2)./NUM_ITERS)'
        Average_Regret[2,:] += (r_op3_learner./NUM_ITERS)'
        Squared_Regret[2,:] += ((r_op3_learner.^2)./NUM_ITERS)'
        Average_Regret[3,:] += (r_ps_learner./NUM_ITERS)'
        Squared_Regret[3,:] += ((r_ps_learner.^2)./NUM_ITERS)'
#        Average_Regret[4,:] += (r_ps2_learner./NUM_ITERS)'
#        Squared_Regret[4,:] += (r_ps2_learner.^2./NUM_ITERS)'

    end
    return Average_Regret, Squared_Regret
end

end
