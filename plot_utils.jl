# stuff for making pretty plots
using PyCall
PyDict(pyimport("matplotlib")["rcParams"])["font.sans-serif"] = ["Helvetica"]
using PyPlot
using JLD

# Initializes configuration for PyPlot, not everything works the way it was meant.
function initialize_plots()
#    PyPlot.svg(true)
    linewidth = 1.2 
    PyPlot.rc("text", usetex=true)
    PyPlot.rc("font", family="serif")
    #PyDict(matplotlib["rcParams"])["font.sans-serif"] = ["Helvetica"]
    #PyPlot.rc("font", sans-serif="Helvetica")
    PyPlot.rc("axes", linewidth=linewidth)
    PyPlot.rc("axes", titlesize=10, labelsize=10)
    PyPlot.rc("xtick", labelsize=10)
    PyPlot.rc("xtick.major", width=linewidth/2)
    PyPlot.rc("ytick", labelsize=10)
    PyPlot.rc("ytick.major", width=linewidth/2)
    PyPlot.rc("legend", fontsize=10)
end


function plot_datafile(filename)
    initialize_plots()

    data = load(filename)
    #assume that data is complete.
    means = data["Average_Regret"]
    sigma = data["Squared_Regret"]
    for i = 1:4
        pretty_plot(i, means[i,:], sigma[i,:])
    end
    legend(["CombLinTS", "CombLinUCB", "CombGPUCB", "SeqCombGPUCB"])
end

# Pretty plotting for regret + confidence interval
function pretty_plot(i, means, sigma)

    means = vec(means);
    sigma = vec(sigma)
    PyPlot.figure(2, figsize=(3,2));
    sigma  = sqrt(vec(sigma) - vec(means).^2);
#    means = means[find(means)]
    means = cumsum(means);
#    sigma = sigma[find(means)]

    L = length(means);
    colors = [:red, :orange, :black, :green]
    plot(collect(1:L), means, color=colors[i]);
    fill_between(collect(1:L), means + sigma, means - sigma, color=colors[i], alpha = 0.3);
    xlabel(L"$\mathrm{Iteration}$", fontsize=10)
    ylabel(L"$\mathrm{Regret}$", fontsize=10)
end

function plot_old_data()
     R_K =[  24.5773   37.6038   66.3529   62.8673   87.8417   99.4096  173.231   161.391   173.375   286.704    392.242   394.232
      15.2746   14.0633   21.3789   22.8239   36.3179   49.9275   53.2236   62.6831   63.2047   97.1291   102.73    127.061
       151.853   260.408   264.205   343.848   397.783   524.073   577.729   673.051   745.813   883.424   1024.16   1044.24 ]

    K = 2*[4:15]

    initialize_plots();
    PyPlot.figure(4,figsize=(3,2))
    ax=gca();
    ax[:patch][:set_visible](false);
    plot(K, vec(R_K[1,:]), color = "red");
    plot(K, vec(R_K[2,:]), color = "orange");
    plot(K, vec(R_K[3,:]), color = "black");
    xlabel(L"$\mathrm{Problem\ size}$", fontsize=10)
    ylabel(L"$\mathrm{Average regret }(T = 50)$", fontsize=10);
    legend([L"$\mathrm{Mean-based OP}", L"$\mathrm{UCB-based OP}$", L"$\mathrm{Posterior Sampling}$"]);
end
function plot_OP_solvetimes()
    problemsizes = [4:2:50]
    times = 0.*zeros(length(problemsizes))
    index = 0;
#    for K in problemsizes
#        index+=1
#        problem = initialize_lattice_problem(K);
#        problem.weights = vec(GPR.sample_n(GPR.GaussianProcessEstimate(problem.prior,2), problem.locations'));
#        t = @timed tmp = solve_OP(problem.weights, problem.distances, problem.Budget, problem.n_start, problem.n_stop);
#        times[index] = t[2]
#        println("$K: ", times[index]);
#    end
    k = [4:2:50]
    times = [0.002420089
    0.011295358
    0.048471505
    0.098256805
    0.300514627
    0.431459249
    0.71386857
    1.084084079
    1.616713654
    2.57380616
    3.571207608
    5.859107302
    7.159102843
    10.114903775
    13.80530353
    16.694673954
    23.381478081
    31.630672861
    41.219513932
    53.2158575
    75.211981148
    87.694839696
    109.215243209
    153.841422806]

    # Plot data prettily
    initialize_plots()
    PyPlot.figure(3,figsize=(3,2))
    ax=gca();
    ax[:patch][:set_visible](false);
    semilogy(problemsizes, times);
    xlabel(L"$ $ Problem\ size", fontsize=10)
    ylabel(L"$ $ Solution\ time (s)", fontsize=10);

end
