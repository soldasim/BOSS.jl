include("../example/data.jl")
include("../src/plotting.jl")

searchdir(path, key) = filter_regexp(readdir(path), key)
filter_regexp(strings, key) = filter(x -> occursin(key, x), strings)

function results()
    files = searchdir("./motor/data/", r"data-2022-10-27-")
    param_files = filter_regexp(files, r"_param_")
    semiparam_files = filter_regexp(files, r"_semiparam_")
    nonparam_files = filter_regexp(files, r"_nonparam_")

    results = [
        [load_data("./motor/data/", f) for f in param_files],
        [load_data("./motor/data/", f) for f in semiparam_files],
        [load_data("./motor/data/", f) for f in nonparam_files],
    ]

    p = plot_bsf_boxplots(results; show_plot=false)
    savefig(p, "./motor/plots/plot.png")
end
