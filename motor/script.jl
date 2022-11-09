using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

ID = ARGS[1]
METHOD = ARGS[2]
data_file = "./motor/data/06/data-" * ID

@show Threads.nthreads()
include("motor_problem.jl")
t = @elapsed compare_models(Symbol(METHOD); info=true, save_run_data=true, file=data_file)
println()
@show t
