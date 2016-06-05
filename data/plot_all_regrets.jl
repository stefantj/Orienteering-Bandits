using PyPlot
using JLD
include("../plot_utils.jl")


function combine_datafiles(filenames, savename)
    i = 0
    for fname in filenames
        if(i == 0)
            i+=1
            data = load(fname)
            fnames = [fname];
            all_regrets = data["All_Regrets"]
            save(savename, "filenames", fnames, "Average_Regret", data["Average_Regret"], "T", data["T"], "All_Regrets", all_regrets[:,:,1:data["iters"]], "iters", data["iters"], "problem_size", data["problem_size"])
        else
            combine_data(fname,savename)
        end
    end
end

function combine_data(filename,Cfile)
    d_combined = load(Cfile)

    if(filename in d_combined["filenames"])
        println("Already added this file!")
    else
        fnames = [d_combined["filenames"], filename]

        d_new = load(filename)
        # assume that all this data was created equally.
        if(d_combined["T"] != d_new["T"])
            error("Mismatching problems!")
        end
        if(d_combined["problem_size"] != d_new["problem_size"])
            error("Mismatching problems!")
        end

        new_regrets = d_new["Average_Regret"]
        new_allregrets = d_new["All_Regrets"]
        it = d_new["iters"]
        new_allregrets = new_allregrets[:,:,1:it]

        iters = d_combined["iters"] + d_new["iters"]
        Average_Regret = (d_combined["Average_Regret"].*d_combined["iters"]+it.*new_regrets)./(iters)
        All_Regrets    = cat(3,d_combined["All_Regrets"], new_allregrets)

        save(Cfile, "filenames", fnames, "problem_size", d_combined["problem_size"], "iters", iters, "Average_Regret", Average_Regret, "All_Regrets", All_Regrets, "T", d_combined["T"])
    end
end

function plot_range_datafile(filename, algos)

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




function plot_progress()

    initialize_plots()
    PyPlot.figure(3,figsize=(3,2))

    combine_datafiles(["wVSMyuPI.jld","gfeb5Nhs.jld","RS8nsa2V.jld"],"17x3x3.jld")
    combine_datafiles(["8LTOPvNT.jld","T0UFIaQt.jld"],"20x3x3.jld")
    combine_datafiles(["R8gtTbly.jld","3tsF9jTf.jld"],"25x3x3.jld")

    filenames = ["Uzkabv7K.jld","lhvcZevt.jld","X7Ts9Bur.jld","mY4klynz.jld","R5axUVud.jld","JCk58LHE.jld"]
    filenames = ["KVKa7TRw.jld","7PNsH5PY.jld","upnDXT4E.jld","60GwfAeH.jld","gSSgBXp6.jld","FHHacFYN.jld","iLjdn7RU.jld","BFnAL3Vo.jld","M1zIzdV6.jld","gcDuENXs.jld","lcbXOiyY.jld","Ou00jKDs.jld","17x3x3.jld","20x3x3.jld","qftLWDQB.jld","25x3x3.jld","uK9Zl3QV.jld","2Lnclmi6.jld","rfsbS0bv.jld"]
    notes = ["1x3x3","2x3x3","3x3x3","4x3x3","5x3x3","6x3x3","7x3x3","8x3x3","9x3x3","10x3x3","13x3x3","15x3x3","17x3x3","20x3x3","22x3x3","25x3x3","27x3x3","30x3x3","35x3x3"]
    N=19
    skip = []#3,4,5]

    lower_k = zeros(4,N)
    lowQ_k  = zeros(4,N)
    mean_k  = zeros(4,N)
    highQ_k = zeros(4,N)
    upper_k = zeros(4,N)


    plot_ind = 0;

    for f in filenames
        plot_ind+=1
        if(plot_ind in skip)
            continue
        end
        subplot(3,4,max(1,plot_ind-8))
        cla()

        data = plot_range_datafile(f,[1:4])
        lower_k[:,plot_ind] = data[:,1]
        lowQ_k[:,plot_ind] = data[:,2]
        mean_k[:,plot_ind] = data[:,3]
        highQ_k[:,plot_ind] = data[:,4]
        upper_k[:,plot_ind] = data[:,5]
        text(100,10,notes[plot_ind])
    end 

    subplot(3,4,12)
    cla();
    sizes = [1,2,3,4,5,6,7,8,9,10,13,15,17,20,22,25,27,30,35]
    K = sizes.*8
    plot(K, mean_k[1,:]',color=:red)
    plot(K,mean_k[4,:]',color=:blue)
    return mean_k, K
end





