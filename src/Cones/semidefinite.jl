#=
Copyright 2018, Chris Coey and contributors

TODO describe hermitian complex PSD cone
on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector

row-wise lower triangle (svec space) of positive semidefinite matrix cone
(smat space) W \in S^n : 0 >= eigmin(W)
(see equivalent MathOptInterface PositiveSemidefiniteConeTriangle definition)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO
- eliminate allocations for inverse-finding
- eliminate redundant svec_to_smat calls
=#

RealOrComplexF64 = Union{Float64, ComplexF64}

mutable struct PosSemidef{T <: RealOrComplexF64} <: Cone
    use_dual::Bool
    dim::Int
    side::Int

    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    Hi::Matrix{Float64}
    mat::Matrix{T}

    function PosSemidef{T}(dim::Int, is_dual::Bool) where {T <: RealOrComplexF64}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim # real vector dimension
        if T <: Complex
            side = isqrt(dim) # real lower triangle and imaginary under diagonal
            @assert side^2 == dim
        else
            side = round(Int, sqrt(0.25 + 2.0 * dim) - 0.5) # real lower triangle
            @assert side * (side + 1) / 2 == dim
        end
        cone.side = side
        return cone
    end
end

# default to real
PosSemidef(dim::Int) = PosSemidef{Float64}(dim, false)
PosSemidef(dim::Int, is_dual::Bool) = PosSemidef{Float64}(dim, is_dual)
PosSemidef{T}(dim::Int) where {T <: RealOrComplexF64} = PosSemidef{T}(dim, false)

function setup_data(cone::PosSemidef{T}) where T
    dim = cone.dim
    cone.mat = Matrix{T}(undef, cone.side, cone.side)
    cone.g = Vector{Float64}(undef, dim)
    cone.H = Matrix{Float64}(undef, dim, dim)
    cone.Hi = similar(cone.H)
    return
end

get_nu(cone::PosSemidef) = cone.side

function set_initial_point(arr::AbstractVector{Float64}, cone::PosSemidef{T}) where T
    incr_off = (T <: Complex) ? 2 : 1
    arr .= 0.0
    k = 1
    for i in 1:cone.side, j in 1:i
        if i == j # on diagonal
            arr[k] = 1.0
            k += 1
        else # off diagonal
            k += incr_off
        end
    end
    return arr
end

function check_in_cone(cone::PosSemidef{T}) where T
    mat = cone.mat
    svec_to_smat!(mat, cone.point)
    F = cholesky!(Hermitian(mat), Val(true), check = false)
    if !isposdef(F)
        return false
    end

    inv_mat = Hermitian(inv(F)) # TODO eliminate allocs
    smat_to_svec!(cone.g, transpose(inv_mat)) # TODO avoid doing this twice
    cone.g .*= -1.0

    # set upper triangles of hessian and inverse hessian
    svec_to_smat!(mat, cone.point)
    H = cone.H
    Hi = cone.Hi

    # TODO refactor
    if T <: Complex
        k = 1
        for i in 1:cone.side, j in 1:i
            k2 = 1
            if i == j
                for i2 in 1:cone.side, j2 in 1:i2
                    if i2 == j2
                        H[k2, k] = abs2(inv_mat[i2, i])
                        Hi[k2, k] = abs2(mat[i2, i])
                        k2 += 1
                    else
                        c = rt2 * conj(inv_mat[i2, i]) * inv_mat[j2, j]
                        ci = rt2 * conj(mat[i2, i]) * mat[j2, j]
                        H[k2, k] = real(c)
                        Hi[k2, k] = real(ci)
                        k2 += 1
                        H[k2, k] = imag(c)
                        Hi[k2, k] = imag(ci)
                        k2 += 1
                    end
                    if k2 > k
                        break
                    end
                end
                k += 1
            else
                for i2 in 1:cone.side, j2 in 1:i2
                    if i2 == j2 # TODO try to merge with other XOR condition above
                        c = rt2 * inv_mat[i2, i] * conj(inv_mat[j2, j])
                        ci = rt2 * mat[i2, i] * (j2 >= j ? mat[j2, j] : conj(mat[j2, j]))
                        H[k2, k] = real(c)
                        Hi[k2, k] = real(ci)
                        H[k2, k + 1] = imag(c)
                        Hi[k2, k + 1] = imag(ci)
                        k2 += 1
                    else
                        c = inv_mat[i2, i] * conj(inv_mat[j2, j]) + inv_mat[j2, i] * conj(inv_mat[i2, j])
                        ci = mat[i2, i] * (j2 >= j ? mat[j2, j] : conj(mat[j2, j])) + mat[j2, i] * (i2 >= j ? mat[i2, j] : conj(mat[i2, j]))
                        c2 = conj(inv_mat[i2, i]) * inv_mat[j2, j] - conj(inv_mat[j2, i]) * inv_mat[i2, j]
                        c2i = conj(mat[i2, i]) * (j2 < j ? mat[j2, j] : conj(mat[j2, j])) - conj(mat[j2, i]) * (i2 < j ? mat[i2, j] : conj(mat[i2, j]))
                        H[k2, k] = real(c)
                        Hi[k2, k] = real(ci)
                        H[k2, k + 1] = imag(c)
                        Hi[k2, k + 1] = imag(ci)
                        k2 += 1
                        H[k2, k] = imag(c2)
                        Hi[k2, k] = imag(c2i)
                        H[k2, k + 1] = real(c2)
                        Hi[k2, k + 1] = real(c2i)
                        k2 += 1
                    end
                    if k2 > k
                        break
                    end
                end
                k += 2
            end
        end
    else
        k = 1
        for i in 1:cone.side, j in 1:i
            k2 = 1
            for i2 in 1:cone.side, j2 in 1:i2
                if (i == j) && (i2 == j2)
                    H[k2, k] = abs2(inv_mat[i2, i])
                    Hi[k2, k] = abs2(mat[i2, i])
                elseif (i != j) && (i2 != j2)
                    H[k2, k] = inv_mat[i2, i] * inv_mat[j, j2] + inv_mat[j2, i] * inv_mat[j, i2]
                    Hi[k2, k] = mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2]
                else
                    H[k2, k] = rt2 * inv_mat[i2, i] * inv_mat[j, j2]
                    Hi[k2, k] = rt2 * mat[i2, i] * mat[j, j2]
                end
                if k2 == k
                    break
                end
                k2 += 1
            end
            k += 1
        end
    end

    return true
end

inv_hess(cone::PosSemidef) = Symmetric(cone.Hi, :U)

inv_hess_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::PosSemidef) = mul!(prod, Symmetric(cone.Hi, :U), arr)
