using BOSS
using Test

# load `TuringExt`
using Turing

```
Determines whether parallelization of BOSS is allowed during tests.
```
const PARALLEL_TESTS = true

@testset "BOSS TESTS" verbose=true begin
    include("unit/runtests.jl")
    include("combinatorial/runtests.jl")
end
