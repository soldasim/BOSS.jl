using BOSS
using Documenter

makedocs(sitename="BOSS.jl";
    pages = [
        "index.md",
        "example.md",
    ]
)

deploydocs(;
    repo = "github.com/soldasim/BOSS.jl",
)
