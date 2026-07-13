text = read(joinpath(@__DIR__, "..", "artilce", "article2.tex"), String)

function countmap2(v)
    d = Dict{String,Int}()
    for x in v
        d[x] = get(d, x, 0) + 1
    end
    return d
end

envs = [String(m.captures[1]) for m in eachmatch(r"\\begin\{([^}]+)\}", text)]
ends = [String(m.captures[1]) for m in eachmatch(r"\\end\{([^}]+)\}", text)]
cb = countmap2(envs); ce = countmap2(ends)
allkeys = union(keys(cb), keys(ce))
mismatched = [k for k in allkeys if get(cb, k, 0) != get(ce, k, 0)]
println("Mismatched environments: ", isempty(mismatched) ? "none" : mismatched)

labels = [String(m.captures[1]) for m in eachmatch(r"\\label\{([^}]+)\}", text)]
lc = countmap2(labels)
dupes = [k for (k, v) in lc if v > 1]
println("Duplicate labels: ", isempty(dupes) ? "none" : dupes)

refs = [String(m.captures[1]) for m in eachmatch(r"\\(ref|Cref|cref)\{([^}]+)\}", text)]
refs2 = [String(m.captures[2]) for m in eachmatch(r"\\(ref|Cref|cref)\{([^}]+)\}", text)]
labelset = Set(labels)
dangling = sort(collect(Set([r for r in refs2 if !(r in labelset)])))
println("Dangling refs: ", isempty(dangling) ? "none" : dangling)

cites = Set{String}()
for m in eachmatch(r"\\cite\{([^}]+)\}", text)
    for c in split(m.captures[1], ",")
        push!(cites, String(strip(c)))
    end
end
bibitems = Set([String(m.captures[1]) for m in eachmatch(r"\\bibitem\{([^}]+)\}", text)])
missing_bib = sort(collect(setdiff(cites, bibitems)))
unused_bib = sort(collect(setdiff(bibitems, cites)))
println("Citations missing bibitem: ", isempty(missing_bib) ? "none" : missing_bib)
println("Unused bibitems: ", isempty(unused_bib) ? "none" : unused_bib)

depth = 0
for ch in text
    global depth
    if ch == '{'
        depth += 1
    elseif ch == '}'
        depth -= 1
    end
end
println("Final brace depth (should be 0): ", depth)
