#=
utilities for Pardiso

TODO add to docs following example:
```julia
import Hypatia

ENV["OMP_NUM_THREADS"] = length(Sys.cpu_info())
import Pardiso

syssolver = Hypatia.Solvers.NaiveElimSparseSystemSolver{Float64}(fact_cache = Hypatia.PardisoNonSymCache())
solver = Hypatia.Solvers.Solver{Float64}(syssolver)

include("test/native.jl")
nonnegative1(Float64, solver = solver)
```
=#

mutable struct PardisoNonSymCache{T <: Real} <: SparseNonSymCache{T}
    analyzed::Bool
    pardiso::Pardiso.MKLPardisoSolver
    function PardisoNonSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        cache.pardiso = Pardiso.MKLPardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_NONSYM)
        return cache
    end
end
PardisoNonSymCache{T}() where {T <: Real} = error("PardisoNonSymCache only works with real type Float64")
PardisoNonSymCache() = PardisoNonSymCache{Float64}()

mutable struct PardisoSymCache{T <: Real} <: SparseSymCache{T}
    analyzed::Bool
    pardiso::Pardiso.MKLPardisoSolver
    diag_pert::Float64
    function PardisoSymCache{Float64}(; diag_pert::Float64 = 0.0)
        cache = new{Float64}()
        cache.analyzed = false
        cache.pardiso = Pardiso.MKLPardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_SYM_INDEF)
        cache.diag_pert = diag_pert
        return cache
    end
end
PardisoSymCache{T}(; diag_pert = 0.0) where {T <: Real} = error("PardisoNonSymCache only works with real type Float64")
PardisoSymCache(; diag_pert = 0.0) = PardisoSymCache{Float64}(diag_pert = diag_pert)

PardisoSparseCache = Union{PardisoSymCache{Float64}, PardisoNonSymCache{Float64}}
int_type(::PardisoSparseCache) = Int32

function update_fact(cache::PardisoSparseCache, A::SparseMatrixCSC{Float64, Int32})
    pardiso = cache.pardiso

    if !cache.analyzed
        Pardiso.pardisoinit(pardiso)
        # solve transposed problem (Pardiso accepts CSR matrices)
        Pardiso.fix_iparm!(pardiso, :N)
        Pardiso.set_phase!(pardiso, Pardiso.ANALYSIS)
        Pardiso.pardiso(pardiso, A, Float64[])
        cache.analyzed = true
    end

    Pardiso.set_phase!(pardiso, Pardiso.NUM_FACT)
    Pardiso.pardiso(pardiso, A, Float64[])

    return
end

function inv_prod(cache::PardisoSparseCache, x::Vector{Float64}, A::SparseMatrixCSC{Float64, Int32}, b::Vector{Float64})
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
