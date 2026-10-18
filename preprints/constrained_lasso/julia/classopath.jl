# The constrained-LASSO path algorithm of Gaines, Kim and Zhou (2018), ported from
# `src/classopath.jl` and `src/constrsparsereg.jl` of ConstrainedLasso.jl
# (https://github.com/Hua-Zhou/ConstrainedLasso.jl, last commit 2018-05-30), which
# targets Julia 0.6 and no longer loads.
#
# Copyright (c) 2017: Hua Zhou. MIT "Expat" License, reproduced in LICENSE.classopath.
#
# The port changes syntax, not the algorithm: Julia 0.6 idioms are replaced by their
# Julia 1 equivalents (find -> findall, countnz -> count, eye -> I, qrfact -> qr,
# scalar assignment into a masked slice -> broadcasting, warn -> @warn), and the
# Convex.jl calls follow its current API (solve! takes an optimizer; the duals are read
# off the constraint objects themselves rather than by inspecting `problem.constraints`).
# Every tolerance, threshold and branch is the original's. Four departures, all marked
# PORT below:
#   * Convex.jl now reports duals with the opposite sign, so they are negated on reading;
#   * Julia 0.6's findmin skipped NaN, Julia 1's returns it, so NaN events are skipped;
#   * the original has a stray `%` at the start of one comment line (a MATLAB comment
#     left in the translation), which Julia 1 parses as an operator; it is removed;
#   * column vectors kept as p-by-1 matrices are kept as vectors.

module ClassoPath

using LinearAlgebra
using Convex
using ECOS

export lsq_classopath

# The original's default, ECOSSolver(maxit=10e8, verbose=0), in today's interface.
default_solver() = Convex.MOI.OptimizerWithAttributes(ECOS.Optimizer, "maxit" => 10^9, "verbose" => 0)

# PORT: the multipliers in the sign convention the algorithm was written for, with the
# stationarity condition X'(y - Xβ) - Aeq'λ - Aineq'μ = ρ s and μ ≥ 0. Convex.jl 0.16
# reports both kinds of dual with the opposite sign to the 2018 release the original
# called, so they are negated here; nothing else changes.
dual(constr) = -vec(collect(constr.dual))

"""
    lsq_constrsparsereg(X, y, ρ; Aeq, beq, Aineq, bineq, penwt, solver)

The constrained LASSO at one value of ρ, by Convex.jl. Returns β, the optimal value,
and the duals of the equality and inequality rows.
"""
function lsq_constrsparsereg(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    ρ::Number;
    Aeq::AbstractMatrix = zeros(T, 0, size(X, 2)),
    beq::AbstractVector = zeros(T, size(Aeq, 1)),
    Aineq::AbstractMatrix = zeros(T, 0, size(X, 2)),
    bineq::AbstractVector = zeros(T, size(Aineq, 1)),
    penwt::AbstractVector = ones(T, size(X, 2)),
    solver = default_solver(),
) where {T}
    p = size(X, 2)
    β = Variable(p)
    loss = (1 // 2) * sumsquares(y - X * β)
    pen = dot(penwt, abs(β))
    constraints = Constraint[]
    eqconstr = isempty(Aeq) ? nothing : (Aeq * β == beq)
    ineqconstr = isempty(Aineq) ? nothing : (Aineq * β <= bineq)
    eqconstr === nothing || push!(constraints, eqconstr)
    ineqconstr === nothing || push!(constraints, ineqconstr)
    if ρ == typemax(typeof(ρ))
        problem = minimize(pen, constraints)
    elseif ρ ≤ 0
        problem = minimize(loss, constraints)
    else
        problem = minimize(loss + ρ * pen, constraints)
    end
    solve!(problem, solver; silent = true)
    λeq = eqconstr === nothing ? zeros(T, 0) : dual(eqconstr)
    μineq = ineqconstr === nothing ? zeros(T, 0) : dual(ineqconstr)
    return vec(collect(evaluate(β))), problem.optval, λeq, μineq
end

"""
    find_ρmax(X, y; Aeq, beq, Aineq, bineq, penidx, solver)

The largest tuning parameter, where the path starts.
"""
function find_ρmax(
    X::AbstractMatrix,
    y::AbstractArray;
    Aeq::AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    beq::AbstractVector = zeros(eltype(X), size(Aeq, 1)),
    Aineq::AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    bineq::AbstractVector = zeros(eltype(X), size(Aineq, 1)),
    penidx::AbstractVector{Bool} = fill(true, size(X, 2)),
    solver = default_solver(),
)
    p = size(X, 2)
    x = Variable(p)
    constraints = Constraint[]
    eqconstr = isempty(Aeq) ? nothing : (Aeq * x == beq)
    ineqconstr = isempty(Aineq) ? nothing : (Aineq * x <= bineq)
    eqconstr === nothing || push!(constraints, eqconstr)
    ineqconstr === nothing || push!(constraints, ineqconstr)
    problem = minimize(dot(ones(eltype(X), p), abs(x)), constraints)
    solve!(problem, solver; silent = true)
    β = vec(collect(evaluate(x)))

    λeq = eqconstr === nothing ? zeros(eltype(X), 0) : dual(eqconstr)
    μineq = ineqconstr === nothing ? zeros(eltype(X), 0) : dual(ineqconstr)

    if !isempty(μineq)
        μineq[μineq .< 0] .= 0
    end
    setActive = (abs.(β) .> 1e-4) .| (.~penidx)
    β[.!setActive] .= 0

    resid = y - X * β
    subgrad = X' * resid - Aeq' * λeq - Aineq' * μineq
    ρmax, indρmax = findmax(abs.(subgrad))

    return ρmax, indρmax, problem.optval, λeq, μineq
end

# The bordered system of the active coefficients and the binding rows. The original
# builds it inline, identically, in six places; PORT: one function, same matrix.
function bordered(H, Aeq, Aineq, setActive, setIneqBorder)
    activeCoeffs = findall(setActive)
    idxIneqBorder = findall(setIneqBorder)
    nActive, neq, nIneqBorder = length(activeCoeffs), size(Aeq, 1), length(idxIneqBorder)
    M = hcat(H[activeCoeffs, activeCoeffs], Aeq[:, activeCoeffs]',
        Aineq[idxIneqBorder, activeCoeffs]')
    M = vcat(M, zeros(neq + nIneqBorder, size(M, 2)))
    M[(end-neq-nIneqBorder+1):end, 1:nActive] = [Aeq[:, activeCoeffs];
        Aineq[idxIneqBorder, activeCoeffs]]
    return M
end

function direction(M, rhs, dirsgn)
    dir = try
        dirsgn * (M \ rhs)
    catch
        -(pinv(M) * rhs)
    end
    if any(isnan.(dir))
        dir = -(pinv(M) * rhs)
    end
    return dir
end

function dirsubgrad(H, Aeq, Aineq, setActive, setIneqBorder, dir)
    activeCoeffs, inactiveCoeffs = findall(setActive), findall(.!setActive)
    idxIneqBorder = findall(setIneqBorder)
    return -hcat(H[inactiveCoeffs, activeCoeffs], Aeq[:, inactiveCoeffs]',
        Aineq[idxIneqBorder, inactiveCoeffs]') * dir
end

inact_slow_neg(subgrad, setActive, dirSubgrad, dirsgn) =
    findall(((1 * dirsgn - 1e-8) .<= subgrad[.~setActive]) .&
            (subgrad[.~setActive] .<= (1 * dirsgn + 1e-8)) .&
            (1 * dirsgn .< dirSubgrad))

inact_slow_pos(subgrad, setActive, dirSubgrad, dirsgn) =
    findall(((-1 * dirsgn - 1e-8) .<= subgrad[.~setActive]) .&
            (subgrad[.~setActive] .<= (-1 * dirsgn + 1e-8)) .&
            (dirSubgrad .< -1 * dirsgn))

sign_mismatch_pos(subgrad, setActive, dir, nActive, βprev, dirsgn) =
    findall(((0 - 1e-8) .<= subgrad[setActive]) .&
            (subgrad[setActive] .<= (1 + 1e-8)) .&
            (dirsgn * dir[1:nActive] .<= (0 - 1e-8)) .&
            (βprev[findall(setActive)] .== 0))

sign_mismatch_neg(subgrad, setActive, dir, nActive, βprev, dirsgn) =
    findall(((-1 - 1e-8) .<= subgrad[setActive]) .&
            (subgrad[setActive] .<= (0 + 1e-8)) .&
            ((0 + 1e-8) .<= dirsgn * dir[1:nActive]) .&
            (βprev[findall(setActive)] .== 0))

"""
    lsq_classopath(X, y; Aeq, beq, Aineq, bineq, ρridge, penidx, solver)

The solution path of `0.5‖y - Xβ‖² + ρ‖β‖₁` subject to `Aeq β = beq`, `Aineq β ≤ bineq`.
Returns βpath (p × knots), ρpath, objvalpath, λpatheq, μpathineq, dfpath, violationspath.

`start`, an addition to the original, is `nothing` (find the start as the original does)
or a named tuple `(ρ, idx, λeq)`: the largest tuning parameter, the coefficient that
enters there, and the multipliers of the equality rows, with β = 0 at that knot.
"""
function lsq_classopath(
    X::AbstractMatrix{T},
    y::AbstractArray{T};
    Aeq::AbstractMatrix = zeros(T, 0, size(X, 2)),
    beq::AbstractVector = zeros(T, size(Aeq, 1)),
    Aineq::AbstractMatrix = zeros(T, 0, size(X, 2)),
    bineq::AbstractVector = zeros(T, size(Aineq, 1)),
    ρridge::Number = zero(T),
    penidx::AbstractVector{Bool} = fill(true, size(X, 2)),
    solver = default_solver(),
    start = nothing,
) where {T}

    n, p = size(X)

    if n < p
        @warn "Adding a small ridge penalty (default is 1e-4) since n < p"
        if ρridge <= 0
            @warn "ρridge must be positive, switching to default value (1e-4)"
            ρridge = 1e-4
        end
        y = [y; zeros(p)]
        X = [X; √ρridge * Matrix{T}(I, p, p)]
    else
        # make sure X is full column rank
        R = qr(X).R
        rankX = sum(abs.(diag(R)) .> abs(R[1, 1]) * max(n, p) * eps(eps(eltype(R))))
        if rankX != p
            @warn "Adding a small ridge penalty (default is 1e-4) since X is rank deficient"
            if ρridge <= 0
                @warn "ρridge must be positive, switching to default value (1e-4)"
                ρridge = 1e-4
            end
            y = [y; zeros(p)]
            X = [X; √ρridge * Matrix{T}(I, p, p)]
        end
    end

    # allocate variables along path
    neq, nineq = size(Aeq, 1), size(Aineq, 1)
    maxiters = 5(p + nineq)
    βpath = zeros(p, maxiters)
    λpatheq = zeros(neq, maxiters)
    μpathineq = zeros(nineq, maxiters)
    ρpath = zeros(maxiters)
    dfpath = fill(Inf, maxiters)
    objvalpath = zeros(maxiters)
    violationspath = fill(Inf, maxiters)

    ### initialization
    H = X' * X

    if start === nothing
        ρpath[1], idx, = find_ρmax(X, y; Aeq = Aeq, beq = beq, Aineq = Aineq,
            bineq = bineq, solver = solver)

        βpath[:, 1], objvalpath[1], λeq, μineq = lsq_constrsparsereg(X, y, ρpath[1];
            Aeq = Aeq, beq = beq, Aineq = Aineq, bineq = bineq,
            penwt = penidx, solver = solver)
    else
        # ADDITION, not in the original: start the walk from a given first knot, with β = 0
        # there, instead of from find_ρmax. The default, `start = nothing`, is the original.
        ρpath[1], idx, λeq, μineq = start.ρ, start.idx, start.λeq, zeros(T, nineq)
    end
    neq > 0 && (λpatheq[:, 1] = λeq)
    nineq > 0 && (μpathineq[:, 1] = μineq)

    μpathineq[μpathineq .< 0] .= 0
    setActive = (abs.(βpath[:, 1]) .> 1e-4) .| (.~penidx)
    βpath[.!setActive, 1] .= 0

    residIneq = Aineq * βpath[:, 1] - bineq
    setIneqBorder = residIneq .== 0
    nIneqBorder = count(setIneqBorder)

    # initialize subgradient vector
    resid = y - X * βpath[:, 1]
    subgrad = X' * resid - Aeq' * λpatheq[:, 1] - Aineq' * μpathineq[:, 1]

    subgrad[setActive] = sign.(βpath[setActive, 1])
    subgrad[.!setActive] = subgrad[.!setActive] / ρpath[1]
    setActive[idx] = true
    nActive = count(setActive)

    rankAeq = rank(Aeq)
    dfpath[1] = nActive - rankAeq - nIneqBorder
    violationspath[1] = 0

    dirsgn = -1

    k = 0
    for outer k in 2:maxiters

        if ρpath[k-1] <= 1e-4
            ρpath[k-1] = 0
            break
        end

        nIneqBorder = count(setIneqBorder)
        M = bordered(H, Aeq, Aineq, setActive, setIneqBorder)
        dir = direction(M, [subgrad[setActive]; zeros(neq + nIneqBorder)], dirsgn)
        dirSubgrad = dirsubgrad(H, Aeq, Aineq, setActive, setIneqBorder, dir)

        inactSlowNegIdx = inact_slow_neg(subgrad, setActive, dirSubgrad, dirsgn)
        inactSlowPosIdx = inact_slow_pos(subgrad, setActive, dirSubgrad, dirsgn)
        signMismatchPosIdx = sign_mismatch_pos(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
        signMismatchNegIdx = sign_mismatch_neg(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)

        violateCounter = 0

        # recompute everything after the active set changed, as the original does
        function refresh!()
            nActive = count(setActive)
            nIneqBorder = count(setIneqBorder)
            M = bordered(H, Aeq, Aineq, setActive, setIneqBorder)
            dir = direction(M, [subgrad[setActive]; zeros(neq + nIneqBorder)], dirsgn)
            dirSubgrad = dirsubgrad(H, Aeq, Aineq, setActive, setIneqBorder, dir)
        end

        while (!isempty(inactSlowNegIdx) || !isempty(inactSlowPosIdx) ||
               !isempty(signMismatchPosIdx) || !isempty(signMismatchNegIdx))

            # condition 1
            while !isempty(inactSlowNegIdx)
                setActive[findall(.!setActive)[inactSlowNegIdx]] .= true
                refresh!()
                inactSlowNegIdx = inact_slow_neg(subgrad, setActive, dirSubgrad, dirsgn)
                inactSlowPosIdx = inact_slow_pos(subgrad, setActive, dirSubgrad, dirsgn)
                signMismatchPosIdx = sign_mismatch_pos(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
                signMismatchNegIdx = sign_mismatch_neg(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
                violateCounter += 1
                violateCounter >= maxiters && break
            end

            # condition 2
            while !isempty(inactSlowPosIdx)
                setActive[findall(.!setActive)[inactSlowPosIdx]] .= true
                refresh!()
                inactSlowPosIdx = inact_slow_pos(subgrad, setActive, dirSubgrad, dirsgn)
                signMismatchPosIdx = sign_mismatch_pos(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
                signMismatchNegIdx = sign_mismatch_neg(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
                violateCounter += 1
                violateCounter >= maxiters && break
            end

            # condition 3
            while !isempty(signMismatchPosIdx)
                setActive[findall(setActive)[signMismatchPosIdx]] .= false
                refresh!()
                signMismatchPosIdx = sign_mismatch_pos(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
                signMismatchNegIdx = sign_mismatch_neg(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
                violateCounter += 1
                violateCounter >= maxiters && break
            end

            # condition 4
            while !isempty(signMismatchNegIdx)
                setActive[findall(setActive)[signMismatchNegIdx]] .= false
                refresh!()
                signMismatchNegIdx = sign_mismatch_neg(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
                violateCounter += 1
                violateCounter >= maxiters && break
            end

            # PORT: the original's stray `%` before the second of these is removed
            inactSlowNegIdx = inact_slow_neg(subgrad, setActive, dirSubgrad, dirsgn)
            inactSlowPosIdx = inact_slow_pos(subgrad, setActive, dirSubgrad, dirsgn)
            signMismatchPosIdx = sign_mismatch_pos(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)
            signMismatchNegIdx = sign_mismatch_neg(subgrad, setActive, dir, nActive, βpath[:, k-1], dirsgn)

            violateCounter >= maxiters && break
        end

        violationspath[k] = violateCounter

        activeCoeffs = findall(setActive)
        inactiveCoeffs = findall(.!setActive)
        idxIneqBorder = findall(setIneqBorder)

        dirResidIneq = Aineq[findall(.!setIneqBorder), activeCoeffs] * dir[1:nActive]

        ### next event
        nextρβ = fill(Inf, p)
        nextρβ[setActive] = -dirsgn * βpath[activeCoeffs, k-1] ./ dir[1:nActive]

        t1 = dirsgn * ρpath[k-1] * (1 .- subgrad[inactiveCoeffs]) ./ (dirSubgrad .- 1)
        t1[t1 .<= 1e-8] .= Inf
        t2 = -dirsgn * ρpath[k-1] * (1 .+ subgrad[.!setActive]) ./ (dirSubgrad .+ 1)
        t2[t2 .<= 1e-8] .= Inf
        nextρβ[.!setActive] = min.(t1, t2)
        nextρβ[(nextρβ .<= 1e-8) .| (.!penidx)] .= Inf

        nextρIneq = fill(Inf, nineq)
        nextρIneq[.!setIneqBorder] = -dirsgn * residIneq[.!setIneqBorder] ./ dirResidIneq
        if !isempty(μpathineq)
            nextρIneq[setIneqBorder] = -dirsgn * μpathineq[idxIneqBorder, k-1] ./
                                       dir[nActive+neq+1:end]
        end
        nextρIneq[nextρIneq .<= 1e-8] .= Inf

        # PORT: Julia 0.6's findmin skipped NaN entries (a 0/0 event); Julia 1 returns NaN
        # as the minimum, so NaN is read as no event, as it was when the code was written.
        events = [nextρβ; nextρIneq]
        chgρ = minimum(e -> isnan(e) ? Inf : e, events; init = Inf)
        idx = findall((events .- chgρ) .<= 1e-8)

        if chgρ == Inf
            chgρ = ρpath[k-1]
        end
        if ρpath[k-1] + dirsgn * chgρ < 0
            chgρ = ρpath[k-1]
        end
        ρpath[k] = ρpath[k-1] + dirsgn * chgρ

        activeCoeffs = findall(setActive)
        βpath[activeCoeffs, k] = βpath[activeCoeffs, k-1] + dirsgn * chgρ * dir[1:nActive]
        βpath[abs.(βpath[:, k]) .< 1e-12, k] .= 0

        subgrad[.!setActive] = (ρpath[k-1] * subgrad[.!setActive] +
                                dirsgn * chgρ * dirSubgrad) / ρpath[k]

        if !isempty(λpatheq)
            λpatheq[:, k] = λpatheq[:, k-1] + dirsgn * chgρ * dir[nActive+1:nActive+neq]
        end
        if !isempty(μpathineq)
            μpathineq[idxIneqBorder, k] = μpathineq[idxIneqBorder, k-1] +
                                          dirsgn * chgρ * dir[nActive+neq+1:end]
        end
        residIneq = Aineq * βpath[:, k] - bineq

        for curidx in idx
            if curidx <= p && setActive[curidx]
                setActive[curidx] = false
            elseif curidx <= p && !setActive[curidx]
                setActive[curidx] = true
            elseif curidx > p
                setIneqBorder[curidx-p] = !setIneqBorder[curidx-p]
            end
        end

        nActive = count(setActive)
        nIneqBorder = count(setIneqBorder)

        objvalpath[k] = norm(y - X * βpath[:, k])^2 / 2 + ρpath[k] * sum(abs.(βpath[:, k]))

        dfpath[k] = nActive - rankAeq - nIneqBorder
        if dfpath[k] >= n
            break
        end
    end

    βpath = βpath[:, 1:k-1]
    deleteat!(ρpath, k:length(ρpath))
    deleteat!(objvalpath, k:length(objvalpath))
    deleteat!(dfpath, k:length(dfpath))
    dfpath[dfpath .< 0] .= 0

    return βpath, ρpath, objvalpath, λpatheq, μpathineq, dfpath, violationspath
end

end
