#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

TODO handle conditional dependencies / glue code, see https://github.com/JuliaLang/Pkg.jl/issues/1285
=#

mutable struct PardisoNonSymCache{T <: Real} <: SparseNonSymCache{T}
    analyzed::Bool
    pardiso::Pardiso.PardisoSolver
    function PardisoNonSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        cache.pardiso = Pardiso.PardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_NONSYM) # tell Pardiso the matrix is nonsymmetric
        return cache
    end
end
PardisoNonSymCache{T}() where {T <: Real} = error("Pardiso only works with real type Float64")
PardisoNonSymCache() = PardisoNonSymCache{Float64}()

mutable struct PardisoSymCache{T <: Real} <: SparseSymCache{T}
    analyzed::Bool
    pardiso::Pardiso.PardisoSolver
    diag_pert::Float64
    function PardisoSymCache{Float64}(; diag_pert::Float64 = 0.0)
        cache = new{Float64}()
        cache.analyzed = false
        cache.pardiso = Pardiso.PardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_SYM_INDEF) # tell Pardiso the matrix is symmetric indefinite
        cache.diag_pert = diag_pert
        return cache
    end
end
PardisoSymCache{T}(; diag_pert = 0.0) where {T <: Real} = error("Pardiso only works with real type Float64")
PardisoSymCache(; diag_pert = 0.0) = PardisoSymCache{Float64}(diag_pert = diag_pert)

PardisoSparseCache = Union{PardisoSymCache{Float64}, PardisoNonSymCache{Float64}}
int_type(::PardisoSparseCache) = Int32

function update_fact(cache::PardisoSparseCache, A::SparseMatrixCSC{Float64, Int32})
    pardiso = cache.pardiso

    if !cache.analyzed
        Pardiso.pardisoinit(pardiso)
        # don't ignore other iparms
        Pardiso.set_iparm!(pardiso, 1, 1)
        # solve transposed problem (Pardiso accepts CSR matrices)
        Pardiso.set_iparm!(pardiso, 12, 1)
        # perturbation for small pivots (default 8 for symmetric, 13 for nonsymmetric)
        if Pardiso.get_matrixtype(pardiso) == Pardiso.REAL_SYM_INDEF
            Pardiso.set_iparm!(pardiso, 10, 8)
        end
        # maximum number of iterative refinement steps (default = 2)
        Pardiso.set_iparm!(pardiso, 8, 2)
        Pardiso.set_phase!(pardiso, Pardiso.ANALYSIS)
        Pardiso.pardiso(pardiso, A, Float64[])
        cache.analyzed = true
    end

    Pardiso.set_phase!(pardiso, Pardiso.NUM_FACT)
    Pardiso.pardiso(pardiso, A, Float64[])

    return
end

function solve_system(cache::PardisoSparseCache, x::VecOrMat{Float64}, A::SparseMatrixCSC{Float64, Int32}, b::VecOrMat{Float64})
    pardiso = cache.pardiso

    Pardiso.set_phase!(pardiso, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(pardiso, x, A, b)

    return x
end

function free_memory(cache::PardisoSparseCache)
    Pardiso.set_phase!(cache.pardiso, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(cache.pardiso)
    return
end
