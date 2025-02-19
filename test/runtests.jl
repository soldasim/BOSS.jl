using BOSS
using Test
using Aqua

# load `TuringExt`
using Turing

```
Determines whether parallelization of BOSS is allowed during tests.
```
const PARALLEL_TESTS = true

@testset "BOSS TESTS" verbose=true begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(BOSS)
    end

    include("unit/runtests.jl")

    include("combinatorial/runtests.jl")
end
