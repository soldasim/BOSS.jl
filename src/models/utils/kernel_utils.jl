
struct CustomKernel <: Kernel
    f::Function
end

(k::CustomKernel)(x, y) = k.f(x, y)
