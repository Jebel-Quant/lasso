# run: after figures
#
# Compare the path algorithm of Gaines, Kim and Zhou (2018) with the exact paths of the
# note.
#
# Their algorithm ships in ConstrainedLasso.jl, which targets Julia 0.6 and no longer
# loads; classopath.jl next to this file is a port of it to Julia 1, with the changes
# listed at its top. make_figures.py writes every problem of Table 2 and its exact path
# to figures/table_paths.npz; this script runs the ported `lsq_classopath` on each, as
# shipped, measures how far its knots lie from the exact path, and writes the numbers
# the note quotes to figures/gaines.tex.
#
# As shipped, the algorithm finds its first knot by a linear program that does not see
# y, then solves the problem there with an interior-point solver and calls a coefficient
# active when it exceeds 1e-4. With homogeneous rows such as sum(beta) = 0 the first step
# returns the wrong lambda; on the diabetes data, whose loss is of order 1e6, the
# solver's tolerance leaves coefficients above 1e-4 that should be zero. Either way the
# walk starts off the path and stays off it. The script therefore also runs, where the
# first knot is simple to state, from the exact first knot, so the path-following can be
# judged on its own.
#
# The line `# run: after figures` above makes `make figures` run this script after
# make_figures.py, in the environment of Project.toml and Manifest.toml, here and in CI.
#
#     julia --project=julia julia/gaines_check.jl

include(joinpath(@__DIR__, "classopath.jl"))
using .ClassoPath
using LinearAlgebra, Logging, NPZ, Printf

const FIGURES = joinpath(dirname(@__DIR__), "figures")
const REPEATS = 5  # each run timed this often, and the fastest counts

"""The problem `name` from the archive, with its bounds written as inequality rows after its own."""
function problem(data, name)
    x, y = data["$name.x"], data["$name.y"]
    p = size(x, 2)
    a, b = get(data, "$name.a", zeros(0, p)), get(data, "$name.b", zeros(0))
    c, h = reshape(get(data, "$name.c", zeros(0, p)), :, p), vec(get(data, "$name.h", zeros(0)))
    lower, upper = data["$name.lower"], data["$name.upper"]
    rows, rhs = Vector{Float64}[eachrow(c)...], Float64[h...]
    for j in 1:p
        isfinite(upper[j]) && (push!(rows, [i == j ? 1.0 : 0.0 for i in 1:p]); push!(rhs, upper[j]))
        isfinite(lower[j]) && (push!(rows, [i == j ? -1.0 : 0.0 for i in 1:p]); push!(rhs, -lower[j]))
    end
    g = isempty(rows) ? zeros(0, p) : permutedims(reduce(hcat, rows))
    return (x = x, y = y, a = reshape(a, :, p), b = vec(b), g = g, h = rhs,
            lam = vec(data["$name.lam"]), beta = data["$name.beta"])
end

"""A path given by knots, affine between them and constant beyond its two ends."""
function at(knots::AbstractVector, betas::AbstractMatrix, l::Real)
    order = sortperm(knots)
    k, b = knots[order], betas[order, :]
    l <= k[1] && return b[1, :]
    l >= k[end] && return b[end, :]
    i = searchsortedlast(k, l)
    s = (l - k[i]) / (k[i+1] - k[i])
    return b[i, :] + s * (b[i+1, :] - b[i, :])
end

"""Largest distance between two paths at the knots of either, relative to the exact path's scale."""
function gap(knots, betas, exact_knots, exact)
    scale = maximum(abs.(exact))
    d = maximum(maximum(abs.(at(knots, betas, l) - at(exact_knots, exact, l))) for l in [knots; exact_knots])
    return d / scale
end

"""Run `lsq_classopath` on a problem, timed; its knots as (ρ, β) with one row per knot."""
function run(prob; start = nothing)
    best, out = Inf, nothing
    for _ in 1:REPEATS
        # silenced: it warns on every call when n < p, and main reports that once
        t = @elapsed (out = with_logger(NullLogger()) do
            lsq_classopath(prob.x, prob.y; Aeq = prob.a, beq = prob.b,
                Aineq = prob.g, bineq = prob.h, start = start)
        end)
        best = min(best, t)
    end
    β, ρ = out[1], out[2]
    return ρ, permutedims(β), best
end

"""The exact first knot, where it is simple to state, else nothing.

Without equality rows, and with no inequality row binding at beta = 0 (h > 0, as for a
cap on |beta_j|), the LASSO leaves zero at max|c|, c = X'y, with the most correlated
coefficient. Under sum(beta) = 0 and no inequality rows it leaves zero at
(max c - min c) / 2, with the multiplier of the row at the midpoint (max c + min c) / 2.
Elsewhere (sign constraints bind at beta = 0, a nonzero right-hand side starts away from
zero) the first knot's multipliers are not unique, and no start is given.
"""
function exact_start(prob)
    c = prob.x' * prob.y
    if isempty(prob.a) && all(prob.h .> 0)
        return (ρ = maximum(abs.(c)), idx = argmax(abs.(c)), λeq = zeros(0))
    end
    isempty(prob.g) || return nothing
    if isempty(prob.a)
        return (ρ = maximum(abs.(c)), idx = argmax(abs.(c)), λeq = zeros(0))
    elseif size(prob.a, 1) == 1 && all(prob.a .== 1) && all(prob.b .== 0)
        return (ρ = (maximum(c) - minimum(c)) / 2, idx = argmax(c), λeq = [(maximum(c) + minimum(c)) / 2])
    end
    return nothing
end

# A count as the note writes it: in words up to ten.
words(n) = n <= 10 ? ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n] : string(n)

sci(x) = (m = @sprintf("%.1e", x); (mant, ex) = split(m, "e"); "\$$(mant)\\times10^{$(parse(Int, ex))}\$")

"""How many coefficients the shipped start calls active at its first knot, where all are zero."""
function spurious(prob)
    ρ, = ClassoPath.find_ρmax(prob.x, prob.y; Aeq = prob.a, beq = prob.b, Aineq = prob.g, bineq = prob.h)
    β, = ClassoPath.lsq_constrsparsereg(prob.x, prob.y, ρ; Aeq = prob.a, beq = prob.b,
        Aineq = prob.g, bineq = prob.h)
    return count(abs.(β) .> 1e-4)
end

function main()
    data = npzread(joinpath(FIGURES, "table_paths.npz"))
    names = sort(unique(first.(split.(collect(keys(data)), "."))))
    shipped, started, starts = Dict{String,Float64}(), Dict{String,Float64}(), Dict{String,Tuple{Float64,Float64}}()
    for name in names
        prob = problem(data, name)
        ρ, β, seconds = try
            run(prob)
        catch err
            @printf("%-16s ConstrainedLasso.jl fails: %s\n", name, sprint(showerror, err))
            continue
        end
        shipped[name] = gap(ρ, β, prob.lam, prob.beta)
        starts[name] = (ρ[1], prob.lam[1])
        @printf("%-16s %3d knots (exact %3d) in %6.1f ms, first at %9.3f (exact %9.3f), largest gap %.1e%s\n",
            name, length(ρ), length(prob.lam), 1e3seconds, ρ[1], prob.lam[1], shipped[name],
            size(prob.x, 1) < size(prob.x, 2) ? "; n < p, so it adds a ridge of 1e-4" : "")
        start = exact_start(prob)
        if start !== nothing
            ρ, β, _ = run(prob; start = start)
            started[name] = gap(ρ, β, prob.lam, prob.beta)
            @printf("%-16s from the exact first knot: %3d knots, largest gap %.1e\n", name, length(ρ), started[name])
        end
    end
    wide = Set(n for n in names if size(data["$n.x"], 1) < size(data["$n.x"], 2))
    exact = filter(n -> haskey(started, n) && !(n in wide), names)
    # As shipped every path starts off the exact one and stays off it; from the exact
    # first knot the walk is on it, except where n < p and the algorithm adds its ridge.
    # Flag it if any of this ever changes.
    for n in keys(shipped)
        @assert shipped[n] > 1e-3 (n, shipped[n])
    end
    for n in exact
        @assert started[n] < 1e-10 (n, started[n])
    end
    for n in wide
        @assert started[n] > 1e-3 (n, started[n])
    end
    # On the diabetes data the shipped start leaves coefficients above its 1e-4 threshold
    # that should be zero: count them, as the note quotes the number.
    diabetes = problem(data, "plain")
    extra = spurious(diabetes)
    @printf("plain: the shipped start calls %d of %d coefficients active at its first knot\n",
        extra, size(diabetes.x, 2))
    @assert extra >= 2 extra
    ex = "zero-sum"
    macros = [
        "gaSpurious" => words(extra),
        "gaCoeffs" => words(size(diabetes.x, 2)),
        "gaStart" => @sprintf("%.1f", starts[ex][1]),          # where the zero-sum example starts
        "gaLamMax" => @sprintf("%.1f", starts[ex][2]),         # where it should
        "gaShippedMin" => sci(minimum(values(shipped))),       # the closest any path comes, as shipped
        "gaShippedMax" => sci(maximum(values(shipped))),       # the farthest
        "gaProblems" => words(length(shipped)),
        "gaStarted" => sci(maximum(started[n] for n in exact)),  # from the exact first knot
        "gaStartedCount" => words(length(exact)),
        "gaWide" => sci(maximum(started[n] for n in wide)),
    ]
    open(joinpath(FIGURES, "gaines.tex"), "w") do io
        for (k, v) in macros
            println(io, "\\newcommand{\\$k}{$v}")
        end
    end
end

main()
