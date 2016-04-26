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
    prior = GPR.GaussianProcess( 0.1, GPR.SquaredExponential(1/pts_dim))

    if( GPR.covar(prior.kernel, 0., 1/(pts_dim-1)) > 0.9)
        warn("Likely unstable covariance matrix: Try adjusting the bandwidth to be smaller");
    end

    weights = zeros(num_pts);

    return BanditProblem(G, locations, distances, prior, weights, B, n_s, n_t)
end

function initialize_trench_problem(pts_dim)

    width=3;
    num_pts = int(pts_dim*width)
    G = simple_graph(num_pts);
    edge_index = 0;
    node_index = 0;
    locs_x = linspace(0,1,pts_dim);
    locs_y = linspace(0,1,width);

    locations = zeros(num_pts,2);

    for i = 1:pts_dim
        for j = 1:width
            node_index +=1
            locations[node_index,:] = [locs_x[i] locs_x[j]];
            if(j < width)
                edge_index+=1;
                add_edge!(G, Edge(edge_index, node_index, node_index+1))
            end
            if(i < pts_dim)
                edge_index+=1;
                add_edge!(G, Edge(edge_index, node_index, node_index+width))
            end
        end
    end

    B = 2.19999;
    distances = 2*B*ones(num_pts,num_pts);
    for E in G.edges
        distances[E.source, E.target] = norm(locations[E.source,:] - locations[E.target,:]);
        if(true || !G.is_directed)
            distances[E.target,E.source] = distances[E.source, E.target]
        end
    end

    # Traverse diagonally
    n_s = 1;
    n_t = num_pts;
    # Initialize prior
    prior = GPR.GaussianProcess( 0.1, GPR.SquaredExponential(.1/pts_dim))

    if( GPR.covar(prior.kernel, 0., 1/(pts_dim-1)) > 0.9)
        warn("Likely unstable covariance matrix: Try adjusting the bandwidth to be smaller");
    end

    weights = zeros(num_pts);

    return BanditProblem(G, locations, distances, prior, weights, B, n_s, n_t)
end
    
function solve_OP(problem::BanditProblem)
    return solve_OP(problem.weights, problem.distances, problem.budget, problem.n_start, problem.n_stop)
end

function solve_OP(values, distances, B,  n_s, n_t; model = Model(solver=Gurobi.GurobiSolver(OutputFlag=0)))
    # Formulate problem for Gurobi:
    N = length(values);
    without_start = [1:n_s-1; n_s+1:N];
    without_stop = [1:n_t-1; n_t+1:N];
    without_both  = intersect(without_start, without_stop);

#    model = Model(solver=Gurobi.GurobiSolver(OutputFlag=0));

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
    reward = zeros(T)
    for t = 1:T
        # Plan action:
        means = GPR.predict_mean(posterior, problem.locations');
        path = solve_OP(means, problem.distances, problem.budget, problem.n_start, problem.n_stop)
        # Now sample the path
        for i = 1:length(path)
            pt = path[i];
            GPR.update(posterior, problem.locations[pt,:]', problem.weights[pt] + sqrt(problem.prior.noise)*randn())
        end
        reward[t] = sum(problem.weights[path]);
    end
    return reward 
end


# Runs the bandit algorithm over a single instance of a problem. Returns the averaged regret over T trials.
function OP2_Bandit(problem::BanditProblem, T::Int64)
    # Extract problem data:
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2)
    reward = zeros(T)
    for t = 1:T
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
            path = solve_OP(ucb, distances, budget_left, path_taken[end], problem.n_stop)
            # In case of failure, see if we can finish the path manually and break.
            if(length(path) == 0) 
         #       println("Fixing path")
         #       if(path_taken[end] != problem.n_stop)
         #           if(problem.distances[path_taken[end], problem.n_stop] <= budget_left)
         #               path_taken = [path_taken, problem.n_stop]
         #           end
         #       end
         #       break
            end
            path_taken = [path_taken, path[2]]
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

# Runs the bandit algorithm over a single instance of a problem. Returns the averaged regret over T trials.
# Uses UCB versus mean to plan. This is so far the best we've done
function OP3_Bandit(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    reward = zeros(T)
    for t = 1:T
        beta_t = sqrt(2*log(t^2*N*(pi^2)/0.6))
        # Plan action:
        ucb = zeros(N);
        for i = 1:N
            mean,var = GPR.predict(posterior, problem.locations[i,:]');
            ucb[i] = mean
            if(!isnan(var))
                ucb[i] += beta_t*sqrt(var);
            else
                ucb[i] += sqrt(2);
            end
        end
        path = solve_OP(ucb, problem.distances, problem.budget, problem.n_start, problem.n_stop)
        # Now sample the path
        for pt in path
            GPR.update(posterior, problem.locations[pt,:]', problem.weights[pt] + sqrt(problem.prior.noise)*randn())
        end
        reward[t] = sum(problem.weights[path]);
    end
    return reward 
end

function PS_Bandit(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    reward = zeros(T)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    for t = 1:T
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

function PS2_Bandit(problem::BanditProblem, T::Int64)
    println("\nResampling posterior bandit going to work...")
    # Extract problem data:
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    distances = deepcopy(problem.distances)
    reward = zeros(T)

    for t = 1:T
        print("-");
    end
    println();
    for t = 1:T
        print("=");
        budget_left = deepcopy(problem.budget)
        path_taken = [problem.n_start];
        while(path_taken[end] != problem.n_stop)

            # Update with sample at current location
            GPR.update(posterior, problem.locations[path_taken[end],:]', problem.weights[path_taken[end]] + sqrt(problem.prior.noise)*randn())
            # Plan action:
            means = real(GPR.sample_n(posterior, problem.locations'))
            path = solve_OP(means, distances, budget_left, path_taken[end], problem.n_stop)
            if(length(path) ==0 )
                break
            end
            path_taken = [path_taken, path[2]]
            budget_left -= problem.distances[path_taken[end-1], path_taken[end]]
            if(length(path_taken) > 1)
                distances[path_taken[end-1],:] = problem.budget
                distances[:,path_taken[end-1]] = problem.budget
            end
        end
        reward[t] =  sum(problem.weights[path_taken])
        for i = 1:length(path_taken)
            pt = path_taken[i];
            GPR.update(posterior, problem.locations[pt,:]', problem.weights[pt] + sqrt(problem.prior.noise)*randn())
        end
    end

    return reward 
end

# Used to return (x,y,z) locations of an example problem
function sample_prior(PROBLEM_SIZE)
    problem_data = initialize_lattice_problem(PROBLEM_SIZE)
    prior = GPR.GaussianProcessEstimate(problem_data.prior,2)
    data = vec(GPR.sample_n(prior, problem_data.locations));

    return problem_data.locations[:,1], problem_data.locations[:,2], data
end

# Test sensitivy for different problem scales
# outdated.

function prior_sensitivity(PROBLEM_SIZE, NUM_ITERS, T_HORIZON)
    problem_data = initialize_lattice_problem(PROBLEM_SIZE);
    ell_values = linspace(0.01,1,100);
    rT = zeros(3,length(ell_values))

    ell_iter = 0;
    for ell in ell_values
        ell_iter+=1;

        print("L = $ell: ");
        problem_data.prior.kernel = GPR.SquaredExponential(ell);
        prior = GPR.GaussianProcessEstimate(problem_data.prior, 2);
        # Test that our matrix will be well conditioned:
        if( GPR.covar(problem_data.prior.kernel, 0., 1/(PROBLEM_SIZE-1)) < 0.9)
            for iteration = 1:NUM_ITERS
                println("==== ITER $iteration ====");
                problem_data.weights = vec(GPR.sample_n(prior, problem_data.locations'));
                optimal_path = solve_OP(problem_data.weights, problem_data.distances, problem_data.budget, problem_data.n_start, problem_data.n_stop);
                s_OP = @spawn OP_Bandit(problem_data, T_HORIZON)
                s_OP3 = @spawn OP3_Bandit(problem_data, T_HORIZON)
                s_PS = @spawn PS_Bandit(problem_data, T_HORIZON)

                r_OP = fetch(s_OP)
                r_OP3 = fetch(s_OP3)
                r_PS = fetch(s_PS)

                rT[1,ell_iter] += sum(r_OP)/NUM_ITERS
                rT[2,ell_iter] += sum(r_OP3)/NUM_ITERS
                rT[3,ell_iter] += sum(r_PS)/NUM_ITERS
            end
        else
            println("Skipped");
        end
    end
    return ell_values, rT
end


function PS_Bandit(problem::BanditProblem, T::Int64)
    N = length(problem.G.vertices)
    reward = zeros(T) 
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
        reward[t] = sum(problem.weights[path])
    end
    return reward
end

function PS2_Bandit(problem::BanditProblem, T::Int64)
    println("\nResampling posterior bandit going to work...")
    # Extract problem data:
    N = length(problem.G.vertices)
    posterior = GPR.GaussianProcessEstimate(problem.prior, 2);
    rewards = zeros(T)

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

    end
    println("ERROR USING UNFINISHED FUNCTION")
        
        # compute regret:

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

## Sample function from the prior: ##
        # TODO:  Consider having this be a separate vector, then problem_data can be immutable (possible speed-up?)
        problem_data.weights = vec(GPR.sample_n(prior, problem_data.locations'))
        optimal_path = solve_OP(problem_data);

## Run learning algorithms: ##
        # Orienteering problem w/o replanning:
        #s_OP = @spawn OP_Bandit(problem_data, T_HORIZON);

        # Orienteering problem w/replanning:
        s_OP2 = @spawn OP2_Bandit(problem_data, T_HORIZON)

        # Orienteering problem w/UCB:
        s_OP3 = @spawn OP3_Bandit(problem_data, T_HORIZON);

        # Posterior sampling w/o replanning:
        s_PS = @spawn PS_Bandit(problem_data, T_HORIZON);

        # Posterior sampling w/replanning:
#        r_ps2_learner = @spawn PS2_Bandit(problem_data,T_HORIZON);


## Record regret: ##
        r_op2_learner = fetch(s_OP2)
        r_op3_learner = fetch(s_OP3)
        r_ps_learner = fetch(s_PS)
        r_opt = sum(problem_data.weights[optimal_path])
        # Find best policy to treat as zero-regret (not needed if optimal_path is computed exactly)
        #r_opt = max(r_opt, 
        #            r_op2_learner[end], 
        #            r_op3_learner[end], 
        #            r_ps_learner[end])


#        Average_Regret[1,:] += (r_op_learner./NUM_ITERS)'
#        Squared_Regret[1,:] += ((r_op_learner.^2)./NUM_ITERS)'
        Average_Regret[1,:] += ((r_opt - r_op2_learner)./NUM_ITERS)'
        Squared_Regret[1,:] += (((r_opt - r_op2_learner).^2)./NUM_ITERS)'
        Average_Regret[2,:] += ((r_opt - r_op3_learner)./NUM_ITERS)'
        Squared_Regret[2,:] += (((r_opt-r_op3_learner).^2)./NUM_ITERS)'
        Average_Regret[3,:] += ((r_opt-r_ps_learner)./NUM_ITERS)'
        Squared_Regret[3,:] += (((r_opt-r_ps_learner).^2)./NUM_ITERS)'
#        Average_Regret[4,:] += (r_ps2_learner./NUM_ITERS)'
#        Squared_Regret[4,:] += (r_ps2_learner.^2./NUM_ITERS)'
    end
    return Average_Regret, Squared_Regret
end

function Bayesian_regret_trench(PROBLEM_SIZE, NUM_ITERS, T_HORIZON)

    problem_data = initialize_trench_problem(PROBLEM_SIZE)
    Average_Regret = zeros(4, T_HORIZON)
    Squared_Regret = zeros(4, T_HORIZON)

    prior = GPR.GaussianProcessEstimate(problem_data.prior,2);
    for iteration = 1:NUM_ITERS
        println("==== ITER $iteration ====");
## Sample function from the prior: ##
        # Consider having this be a separate vector, then problem_data can be immutable (possible speed-up?)
        problem_data.weights = vec(GPR.sample_n(prior, problem_data.locations'))
        optimal_path = solve_OP(problem_data)

## Run learning algorithms: ##
        # Orienteering problem w/o replanning:
#        s_OP = @spawn OP_Bandit(problem_data, T_HORIZON);

        # Orienteering problem w/replanning:
        S_OP2 = @spawn OP2_Bandit(problem_data, T_HORIZON);

        # Orienteering problem w/UCB:
        s_OP3 = @spawn OP3_Bandit(problem_data, T_HORIZON);

        # Posterior sampling w/o replanning:
        s_PS = @spawn PS_Bandit(problem_data, T_HORIZON);

        # Posterior sampling w/replanning:
#        r_ps2_learner = @spawn PS2_Bandit(problem_data,T_HORIZON);

## Record regret: ##
#        r_op_learner = fetch(s_OP)
        r_op2_learner = fetch(S_OP2)
        r_op3_learner = fetch(s_OP3)
        r_ps_learner = fetch(s_PS)
        r_opt = sum(problem_data.weights[optimal_path]);
        # Find best policy to treat as zero-regret (not needed if optimal_path is computed exactly)
        r_opt = max(r_opt,
                    r_op2_learner[end],
                    r_op3_learner[end],
                    r_ps_learner[end])

#        Average_Regret[1,:] += (r_op_learner./NUM_ITERS)'
#        Squared_Regret[1,:] += ((r_op_learner.^2)./NUM_ITERS)'
        Average_Regret[1,:] += ((r_opt - r_op2_learner)./NUM_ITERS)'
        Squared_Regret[1,:] += (((r_opt - r_op2_learner).^2)./NUM_ITERS)'
        Average_Regret[2,:] += ((r_opt - r_op3_learner)./NUM_ITERS)'
        Squared_Regret[2,:] += (((r_opt-r_op3_learner).^2)./NUM_ITERS)'
        Average_Regret[3,:] += ((r_opt-r_ps_learner)./NUM_ITERS)'
        Squared_Regret[3,:] += (((r_opt-r_ps_learner).^2)./NUM_ITERS)'
#        Average_Regret[4,:] += (r_ps2_learner./NUM_ITERS)'
#        Squared_Regret[4,:] += (r_ps2_learner.^2./NUM_ITERS)'

    end
    return Average_Regret, Squared_Regret
end


end
