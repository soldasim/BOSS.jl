
function list_files_rec(directory)
    directory = rstrip(directory, ['\\', '/'])
    files = reduce(vcat, ([dir*'\\'*f for f in files] for (dir, _, files) in walkdir(directory)))
    return files
end
