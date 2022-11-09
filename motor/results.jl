include("../example/data.jl")
include("../src/plotting.jl")

searchdir(path, key) = filter_regexp(readdir(path), key)
filter_regexp(strings, key) = filter(x -> occursin(key, x), strings)

function results()
    dir = "06/"
    datadir = "./motor/data/" * dir
    plotdir = "./motor/plots/" * dir

    files = searchdir(datadir, r"data")
    param_files = reduce(vcat, filter_regexp(files, r"_param_"))
    semiparam_files = reduce(vcat, filter_regexp(files, r"_semiparam_"))
    nonparam_files = reduce(vcat, filter_regexp(files, r"_nonparam_"))

    @show length(param_files)
    @show length(semiparam_files)
    @show length(nonparam_files)

    L = minimum([length(param_files), length(semiparam_files), length(nonparam_files)])
    println("Taking $L first runs from each.")

    results = [
        [load_data(datadir, param_files[i]) for i in 1:L],
        [load_data(datadir, semiparam_files[i]) for i in 1:L],
        [load_data(datadir, nonparam_files[i]) for i in 1:L],
    ]

    # BSF
    p = plot_bsf_boxplots(results; show_plot=false)
    savefig(p, plotdir * "bsf.png")

    # # PARAM
    # param_idx = 1
    # true_val = 0.8
    # p = plot_paramdiff_boxplots(true_val, param_idx, results; show_plot=false)
    # savefig(p, plotdir * "paramdiff.png")
end
