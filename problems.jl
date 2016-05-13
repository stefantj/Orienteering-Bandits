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
    is_DAC::Bool                    # Whether to use Dijkstra's Algorithm
end

# re-samples a given problem structure (for monte-carlo simulations)
function resample_bandit_problem(problem_data::BanditProblem)
    println("Resampling problem")
    data = @timed vec(GPR.sample_n(GPR.GaussianProcessEstimate(problem_data.prior,2), problem_data.locations'))
    new_weights = data[1]
    println(" took ", data[2], " seconds")
    return BanditProblem(problem_data.G, problem_data.locations, problem_data.distances, problem_data.prior, new_weights, problem_data.budget, problem_data.n_start, problem_data.n_stop, problem_data.is_DAC)
end

# Returns a subproblem with only selected nodes. TODO: make more efficient
function subproblem(problem::BanditProblem, selected_nodes)
    num_pts = length(selected_nodes)
    G = simple_graph(num_pts)
    G.is_directed = problem.G.is_directed
    locations = problem.locations[selected_nodes,:]
    distances = problem.distances[selected_nodes, selected_nodes]
    prior = problem.prior
    B = problem.budget


    edge_index = 0
    for edge in problem.G.edges
        if(edge.source in selected_nodes && edge.target in selected_nodes)
            edge_index+=1
            node_index_s = find(selected_nodes .==edge.source)[1]
            node_index_t = find(selected_nodes .==edge.target)[1]
            add_edge!(G, Edge(edge_index, node_index_s, node_index_t))
            if(node_index_s > num_pts || node_index_t > num_pts)
                error("indexing mistake in subproblem")
            end
        end
    end

    n_s=1
    if(!isempty(find(selected_nodes.==problem.n_start)))
        n_s = find(selected_nodes.==problem.n_start)[1]
    end
    n_t = find(selected_nodes.==problem.n_stop)[1]

    return BanditProblem(G, problem.locations[selected_nodes,:], problem.distances[selected_nodes,selected_nodes], problem.prior, problem.weights[selected_nodes], problem.budget,n_s,n_t, problem.is_DAC)
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
    B = 3.0
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
    prior = GPR.GaussianProcess( 0.1, GPR.SquaredExponential(1/pts_dim,1.0))

    if( GPR.covar(prior.kernel, 0., 1/(pts_dim-1)) > 0.9)
        warn("Likely unstable covariance matrix: Try adjusting the bandwidth to be smaller");
    end

    weights = zeros(num_pts);

    return BanditProblem(G, locations, distances, prior, weights, B, n_s, n_t, directed)
end

function initialize_trench_problem(width, pts_dim; directed=true)

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
    prior = GPR.GaussianProcess( 0.1, GPR.SquaredExponential(0.01/pts_dim,1.0))

    if( GPR.covar(prior.kernel, 0., 1/(pts_dim-1)) > 0.9)
        warn("Likely unstable covariance matrix: Try adjusting the bandwidth to be smaller");
    end

    weights = zeros(num_pts);

    return BanditProblem(G, locations, distances, prior, weights, B, n_s, n_t, directed)
end
   
# Loads wind data from datafile.csv
function initialize_wind_problem(datafile)
    d = load(datafile)
    NT = d["NT"]
    NA = d["NA"]
    locations = d["locations"]
    distances = d["distances"]
    prior = d["prior"]
    weights = d["weights"]

    num_pts = NT*NA
    G = simple_graph(num_pts)
    G.is_directed = directed
    edge_index = 0
    node_index = 0
    for t = 1:NT
        for a = 1:NA
            node_index += 1
            if(t < NT)
                # Connect right if t < NT
                edge_index += 1
                add_edge!(G, Edge(edge_index, node_index, node_index + NA))
                # Connect down if not on bottom or edge
                if(a > 1)
                    edge_index += 1
                    add_edge!(G, Edge(edge_index, node_index, node_index + NA - 1))
                end
                # Connect up if not on top or edge
                if(a < NA)
                    edge_index += 1
                    add_edge!(G, Edge(edge_index, node_index, node_index + NA + 1))
                end
            end
        end
    end

    # Budget is irrelevant since DAG
    B = 100000
    n_s = 1
    n_t = num_pts

    return BanditProblem(G, locations, distances, prior, weights, B, n_s, n_t, true)
end

