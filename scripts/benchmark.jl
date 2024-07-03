using BenchmarkTools

include("example.jl")

"""
A simple benchmark showcasing the speed-up achieved via parallelization with MLE.
"""
function parallel_benchmark_mle()
    options = BossOptions(; info=false, plot_options=nothing)
    
    seq_mle = @elapsed example_mle(; parallel=false, options)
    par_mle = @elapsed example_mle(; parallel=true, options)

    println("SEQUENTIAL MLE: $seq_mle")
    println("PARALLEL MLE: $par_mle")
    return seq_mle, par_mle
end

"""
A simple benchmark showcasing the speed-up achieved via parallelization with BI.
"""
function parallel_benchmark_bi()
    options = BossOptions(; info=false, plot_options=nothing)
    
    seq_bi = @elapsed example_bi(; parallel=false, options)
    par_bi = @elapsed example_bi(; parallel=true, options)

    println("SEQUENTIAL BI: $seq_bi")
    println("PARALLEL BI: $par_bi")
    return seq_bi, par_bi
end
