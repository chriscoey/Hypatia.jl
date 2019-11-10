#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see "A Direct Formulation for Sparse PCA Using Semidefinite Programming" by
Alexandre d’Aspremont, Laurent El Ghaoui, Michael I. Jordan, Gert R. G. Lanckriet
==#

using LinearAlgebra
import Distributions
import Random
using Test
import Hypatia
import Hypatia.BlockMatrix
const CO = Hypatia.Cones

function sparsepca(
    T::Type{<:Real},
    p::Int,
    k::Int;
    use_l1ball::Bool = true,
    noise_ratio::Float64 = 0.0,
    use_linops::Bool = false,
    )
    @assert 0 < k <= p

    signal_idxs = Distributions.sample(1:p, k, replace = false) # sample components that will carry the signal
    if noise_ratio <= 0.0
        # noiseless model
        x = zeros(T, p)
        x[signal_idxs] = rand(T, k)
        sigma = x * x'
        sigma ./= tr(sigma)
        true_obj = -1
    else
        # simulate some observations with noise
        x = randn(p, 100)
        sigma = x * x'
        y = rand(Distributions.Normal(0, noise_ratio), k)
        sigma[signal_idxs, signal_idxs] .+= y * y'
        sigma ./= 100
        sigma = T.(sigma)
        true_obj = NaN
    end

    rt2 = sqrt(T(2))
    dimx = div(p * (p + 1), 2)
    # x will be the vec (lower triangle, row-wise) of the matrix solution we seek
    c = CO.smat_to_svec!(zeros(T, dimx), -sigma, rt2)
    b = T[1]
    A = zeros(T, 1, dimx)
    for i in 1:p
        s = sum(1:i)
        A[s] = 1
    end
    hpsd = zeros(T, dimx)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(dimx)]

    if use_l1ball
        # l1 cone
        # double off-diagonals, which are already scaled by rt2
        if use_linops
            Gl1 = Diagonal(-one(T) * I, dimx)
        else
            Gl1 = -Matrix{T}(I, dimx, dimx)
        end
        MU.vec_to_svec_cols!(Gl1, rt2)
        if use_linops
            G = BlockMatrix{T}(
                2 * dimx + 1,
                dimx,
                [-I, Gl1],
                [1:dimx, (dimx + 2):(2 * dimx + 1)],
                [1:dimx, 1:dimx]
                )
            A = BlockMatrix{T}(1, dimx, [A], [1:1], [1:dimx])
        else
            G = [
                Matrix{T}(-I, dimx, dimx); # psd cone
                zeros(T, 1, dimx);
                Gl1;
                ]
        end
        h = vcat(hpsd, T(k), zeros(T, dimx))
        push!(cones, CO.EpiNormInf{T}(1 + dimx, true))
    else
        id = Matrix{T}(I, dimx, dimx)
        l1 = MU.vec_to_svec!(ones(T, dimx), rt2)
        if use_linops
            A = BlockMatrix{T}(
                dimx + 1,
                3 * dimx,
                [A, -I, -I, I],
                [1:1, 2:(dimx + 1), 2:(dimx + 1), 2:(dimx + 1)],
                [1:dimx, 1:dimx, (dimx + 1):(2 * dimx), (2 * dimx + 1):(3 * dimx)]
                )
            G = BlockMatrix{T}(
                3 * dimx + 1,
                3 * dimx,
                [-I, -I, repeat(l1', 1, 2)],
                [1:dimx, (dimx + 1):(3 * dimx), (3 * dimx + 1):(3 * dimx + 1)],
                [1:dimx, (dimx + 1):(3 * dimx), (dimx + 1):(3 * dimx)]
                )
        else
            A = T[
                A    zeros(T, 1, 2 * dimx);
                -id    -id    id;
                ]
            G = [
                Matrix{T}(-I, dimx, dimx)    zeros(T, dimx, 2 * dimx);
                zeros(T, 2 * dimx, dimx)    Matrix{T}(-I, 2 * dimx, 2 * dimx);
                zeros(T, 1, dimx)    repeat(l1', 1, 2);
                ]
        end
        c = vcat(c, zeros(T, 2 * dimx))
        b = vcat(b, zeros(T, dimx))
        h = vcat(hpsd, zeros(T, 2 * dimx), k)
        push!(cones, CO.Nonnegative{T}(2 * dimx + 1))
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, true_obj = true_obj)
end

sparsepca1(T::Type{<:Real}) = sparsepca(T, 5, 3)
sparsepca2(T::Type{<:Real}) = sparsepca(T, 5, 3, use_l1ball = false)
sparsepca3(T::Type{<:Real}) = sparsepca(T, 10, 3)
sparsepca4(T::Type{<:Real}) = sparsepca(T, 10, 3, use_l1ball = false)
sparsepca5(T::Type{<:Real}) = sparsepca(T, 10, 3, noise_ratio = 10.0, use_linops = false)
sparsepca6(T::Type{<:Real}) = sparsepca(T, 10, 3, noise_ratio = 10.0, use_linops = true)
sparsepca7(T::Type{<:Real}) = sparsepca(T, 10, 3, noise_ratio = 10.0, use_l1ball = false, use_linops = false)
sparsepca8(T::Type{<:Real}) = sparsepca(T, 10, 3, noise_ratio = 10.0, use_l1ball = false, use_linops = true)

instances_sparsepca_all = [
    sparsepca1,
    sparsepca2,
    sparsepca3,
    sparsepca4,
    sparsepca5,
    sparsepca7,
    ]
instances_sparsepca_linops = [
    sparsepca5,
    sparsepca6,
    sparsepca7,
    sparsepca8,
    ]
instances_sparsepca_few = [
    sparsepca1,
    sparsepca2,
    ]

function test_sparsepca(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = (atol = sqrt(sqrt(eps(T))),), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    if !isnan(d.true_obj)
        @test r.primal_obj ≈ d.true_obj atol=options.atol rtol=options.atol
    end
    return
end
