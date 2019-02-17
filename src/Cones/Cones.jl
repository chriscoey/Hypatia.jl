#=
Copyright 2018, Chris Coey and contributors

functions and caches for cones
=#

module Cones

using LinearAlgebra
using ForwardDiff
using DiffResults

abstract type Cone end

include("orthant.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("hypoperlog.jl")
include("epiperpower.jl")
include("epipersumexp.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("semidefinite.jl")
include("hypoperlogdet.jl")
include("wsospolyinterp.jl")
include("wsospolyinterpmat.jl")

use_dual(cone::Cone) = cone.use_dual
load_point(cone::Cone, point::AbstractVector{Float64}) = (cone.point = point)
dimension(cone::Cone) = cone.dim

function factorize_hess(cone::Cone)
    @. cone.H2 = cone.H

    cone.F = bunchkaufman!(Symmetric(cone.H2, :U), true, check = false)
    return issuccess(cone.F)

    # cone.F = cholesky!(Symmetric(cone.H2), Val(true), check = false)
    # return isposdef(cone.F)
end

grad(cone::Cone) = cone.g
hess(cone::Cone) = Symmetric(cone.H, :U)
inv_hess(cone::Cone) = inv(cone.F)
# hess_fact(cone::Cone) = cone.F
# hessL(cone::Cone) = cone.F.L
# inv_hessL(cone::Cone) = inv(cone.F.L)
# hess_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::Cone) = mul!(prod, cone.H, arr)
# inv_hess_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::Cone) = ldiv!(prod, cone.F, arr)
# hessL_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::Cone) = mul!(prod, cone.F.L, arr)
# inv_hessL_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::Cone) = ldiv!(prod, cone.F.L, arr)

# utilities for converting between smat and svec forms (lower triangle) for symmetric matrices
# TODO only need to do lower triangle if use symmetric matrix types
const rt2 = sqrt(2)
const rt2i = inv(rt2)

function smat_to_svec!(vec::AbstractVector, mat::AbstractMatrix)
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = rt2 * mat[i, j]
        end
        k += 1
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix, vec::AbstractVector)
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = mat[j, i] = rt2i * vec[k]
        end
        k += 1
    end
    return mat
end

end
