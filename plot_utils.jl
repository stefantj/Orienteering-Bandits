# plot_utils.jl
# stuff for making pretty plots

#This is for getting pyplot to print with LaTex
using PyCall
PyDict(pyimport("matplotlib")["rcParams"])["font.sans-serif"] = ["Helvetica"]
using PyPlot
using JLD

# Initializes configuration for PyPlot, not everything works the way it should. Looks pretty though.
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


# Plots regret for CombLinTS and SeqGPUCB.
# The lightly shaded regions represent the range of all samples, the dark shading is the 1-sigma interval, the solid line is the mean and the dotted line is the median.
function plot_datafile(filename)
    # Set up figure
    figure(1); clf();
    initialize_plots()

    d = load(filename)

    T = d["T"]
    iters = d["iters"]
    top_Q = int(floor(iters*.75))+1
    low_Q = int(floor(iters*.25))+1

    Average_Regret = d["Average_Regret"]
    All_Regrets = d["All_Regrets"]

    upper = zeros(4,T)
    top25 = zeros(4,T)
    bottom25 = zeros(4,T)
    lower = zeros(4,T)
    avg = zeros(4,T)
    med = zeros(4,T)

    # Process data
    algos = [1,4] # This is a concession to old code
     for algo in algos
        for t = 2:T
            for i = 1:iters
                if(isnan(All_Regrets[algo,t,i]))
                    All_Regrets[algo,t,i] = All_Regrets[algo,t-1,i]
                end
                All_Regrets[algo,t,i] += All_Regrets[algo,t-1,i]
            end
        end
    end
    
    for algo in algos
        for t = 1:T
            vals = sort(vec(All_Regrets[algo,t,1:iters]))
            lower[algo,t] = vals[1]
            bottom25[algo,t] = vals[1]
            med[algo,t] = vals[int(floor(iters/2))+1]
            upper[algo,t] = vals[end]
            top25[algo,t] = vals[end]
            avg[algo,t] = sum(vals[1:end])/iters
        end
    end

    diffs = zeros(iters)
    for i = 1:iters
        diffs[i] = sum(All_Regrets[1,:,i]) - sum(All_Regrets[4,:,i])
    end
    sharpe = mean(diffs)/std(diffs)
println(sharpe)

    # Now plot:
    colors = [:red, :green, :black, :blue]

    
    for algo in algos
        plot(collect(1:T), avg[algo,:]', color=colors[algo])
    end
#    legend(["CombLinTS", "CombLinUCB", "GPCombUCB (untuned)", "SeqCombGPUCB (tuned)"], loc="lower right")
    for algo in algos
        plot(collect(1:T), vec(med[algo,:]'), color=colors[algo], linestyle="-.")
        fill_between(collect(1:T), vec(upper[algo,:]'), vec(lower[algo,:]'), color=colors[algo], alpha=0.1)
        fill_between(collect(1:T), vec(top25[algo,:]'), vec(bottom25[algo,:]'), color=colors[algo], alpha=0.4)
    end
    psize = int(sqrt(d["problem_size"]))
    #println("$psize x $psize")
    if(d["problem_size"] < 36)
#        ylim(-1,50)
#        text(0.6*T, 44, "Problem size: $psize x $psize")
#        text(0.6*T, 40, "Samples: $iters")
    elseif(d["problem_size"] < 81)
#        ylim(-1,100)
#        text(0.6*T, 88, "Problem size: $psize x $psize\nSamples: $iters")
#        text(0.6*T, 80, "Samples: $iters")
    else
        ylim(-1,250)
       # text(0.6*T, 3*44, "Problem size: $psize x $psize")
#        text(0.6*T, 5*44, "Problem size: 3 x 40\n Samples: $iters")
#        text(0.1*T, 5*44, "Shaded region shows range of results\n Solid curve shows mean results.\n Dashed curve is median.")
    end


    xlabel(L"$\mathrm{Learning\ time}$", fontsize=20)
    ylabel(L"$\mathrm{Regret}$", fontsize=20)
    if(4 in algos)
        algo=4
        for i = 1:iters
            plot(vec(All_Regrets[algo,:,i]), color=colors[algo], alpha=0.03)
        end
    end


        #println("Median ratio: ", med[4,end]/med[1,end] , " Average ratio: ", avg[4,end]/avg[1,end])

#    return [lower[:,1000] bottom25[:,1000] avg[:,1000] top25[:,1000] upper[:,1000]]
    return [lower[:,end] bottom25[:,end] med[:,end] top25[:,end] upper[:,end]]
end
