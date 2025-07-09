
"""
    CustomKernel(::Function)

Auxiliary structure to define a custom kernel by passing a function:

`(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) -> value::Real`
"""
struct CustomKernel <: Kernel
    f::Function
end

(k::CustomKernel)(x, y) = k.f(x, y)

"""
    DiscreteKernel(kernel::Kernel, dims::AbstractVector{Bool})
    DiscreteKernel(kernel::Kernel)

A kernel for dealing with discrete variables.
It is used as a wrapper around any other `AbstractGPs.Kernel`.

The field `dims` can be used to specify only some dimension as discrete.
All dimensions are considered as discrete if `dims` is not provided.

This structure is used internally by the BOSS algorithm.
The end user of BOSS.jl is not expected to use this structure.
Use the `Domain` passed to the `BossProblem`
to define discrete dimensions instead.

See also: `BossProblem`(@ref)

## Examples:
```julia-repl
julia> BOSS.DiscreteKernel(BOSS.Matern32Kernel())
BOSS.DiscreteKernel{Missing}(Matern 3/2 Kernel (metric = Distances.Euclidean(0.0)), missing)

julia> BOSS.DiscreteKernel(BOSS.Matern32Kernel(), [true, false, false])
BOSS.DiscreteKernel{Vector{Bool}}(Matern 3/2 Kernel (metric = Distances.Euclidean(0.0)), Bool[1, 0, 0])

julia> 
```
"""
@kwdef struct DiscreteKernel{
    D<:Union{Missing, AbstractVector{Bool}}
} <: Kernel
    kernel::Kernel
    dims::D = missing
end

make_discrete(k::Kernel, ::Nothing) = k
make_discrete(k::Kernel, discrete) = DiscreteKernel(k, discrete)

make_discrete(k::DiscreteKernel, ::Nothing) = k.kernel
make_discrete(k::DiscreteKernel, discrete) = DiscreteKernel(k.kernel, discrete)

function (dk::DiscreteKernel)(x1, x2)
    r(x) = discrete_round(dk.dims, x)
    dk.kernel(r(x1), r(x2))
end

KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscale::Real) =
    DiscreteKernel(with_lengthscale(dk.kernel, lengthscale), dk.dims)
KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscales::AbstractVector{<:Real}) =
    DiscreteKernel(with_lengthscale(dk.kernel, lengthscales), dk.dims)

# Necessary to make `DiscreteKernel` work with ForwardDiff.jl.
# See: https://github.com/soldasim/BOSS.jl/issues/4
KernelFunctions.kernelmatrix_diag(dk::DiscreteKernel, x::AbstractVector) =
    kernelmatrix_diag(dk.kernel, discrete_round.(Ref(dk.dims), x))
