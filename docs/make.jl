using BOSS
using Documenter

makedocs(sitename="BOSS.jl";
    pages = [
        "index.md",
        "docs.md",
        "example.md",
    ],
)

deploydocs(;
    repo = "github.com/soldasim/BOSS.jl",
)
