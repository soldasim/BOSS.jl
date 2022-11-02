using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

ID = ARGS[1]
data_file = "./motor/data/01/data-" * ID

@show Threads.nthreads()
include("motor_problem.jl")
t = @elapsed compare_models(; info=true, save_run_data=true, file=data_file)
@show t
