using BOSS
using Documenter

makedocs(sitename="BOSS.jl";
    pages = [
        "index.md",
        "functions.md",
        "types.md",
        "example.md",
    ],
)

deploydocs(;
    repo = "github.com/soldasim/BOSS.jl",
    versions = [
        "stable" => "v^",
    ]
)
