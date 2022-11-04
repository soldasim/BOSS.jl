include("../example/data.jl")
include("../src/plotting.jl")

searchdir(path, key) = filter_regexp(readdir(path), key)
filter_regexp(strings, key) = filter(x -> occursin(key, x), strings)

function results()
    dir = "./motor/data/01/"

    files = searchdir(dir, r"data")
    param_files = reduce(vcat, filter_regexp(files, r"_param_"))
    semiparam_files = reduce(vcat, filter_regexp(files, r"_semiparam_"))
    nonparam_files = reduce(vcat, filter_regexp(files, r"_nonparam_"))

    @show length(param_files)
    @show length(semiparam_files)
    @show length(nonparam_files)

    results = [
        [load_data(dir, f) for f in param_files],
        [load_data(dir, f) for f in semiparam_files],
        [load_data(dir, f) for f in nonparam_files],
    ]

    p = plot_bsf_boxplots(results; show_plot=false)
    savefig(p, "./motor/plots/plot_3.png")
end
