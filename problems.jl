# Contains problem initialization routines for Combinatorial Bandits
export BanditProblem
export initialize_lattice_problem
export initialize_trench_problem


# Contains all of the necessary information for an orienteering bandit problem.
immutable BanditProblem
    G::GenericGraph                 # Node topology
    locations::Array{Float64,2}     # Node locations (in covariance space)
    distances::Array{Float64,2}     # Pairwise distances
    prior::GPR.GaussianProcess      # Prior distribution
    weights::Vector{Float64}        # Function weights
    budget::Float64                 # Travel budget
    n_start::Int64                  # Start node
    n_stop::Int64                   # Stop node
end

# re-samples a given problem structure (for monte-carlo simulations)
function resample_bandit_problem(problem_data::BanditProblem)
    new_weights = vec(GPR.sample_n(GPR.GaussianProcessEstimate(problem_data.prior,2), problem_data.locations'))
    return BanditProblem(problem_data.G, problem_data.locations, problem_data.distances, problem_data.prior, new_weights, problem_data.budget, problem_data.n_start, problem_data.n_stop)
end

# Initializes a problem with lattice graph structure. 
function initialize_lattice_problem(pts_dim; directed = true)

    num_pts = int(pts_dim^2)
    G = simple_graph(num_pts);
    G.is_directed = directed;

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

function initialize_trench_problem(pts_dim; directed=true)

    width=3;
    num_pts = int(pts_dim*width)
    G = simple_graph(num_pts)
    G.is_directed = directed
    edge_index = 0;
    node_index = 0;
    locs_x = linspace(0,1,pts_dim)
    locs_y = linspace(0,1,width)

    locations = zeros(num_pts,2)

    for i = 1:pts_dim
        for j = 1:width
            node_index +=1
            locations[node_index,:] = [locs_x[i] locs_x[j]]
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
    prior = GPR.GaussianProcess( 0.1, GPR.SquaredExponential(0.01/pts_dim))

    if( GPR.covar(prior.kernel, 0., 1/(pts_dim-1)) > 0.9)
        warn("Likely unstable covariance matrix: Try adjusting the bandwidth to be smaller");
    end

    weights = zeros(num_pts);

    return BanditProblem(G, locations, distances, prior, weights, B, n_s, n_t)
end
    

