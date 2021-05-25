#=
helpers for dense factorizations and linear solves
=#

import LinearAlgebra.BlasReal
import LinearAlgebra.BlasFloat
import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.LAPACK.liblapack
import LinearAlgebra.copytri!

# helpers for in-place Cholesky inverse

function chol_inv!(
    mat::Matrix{R},
    fact::Cholesky{R},
    ) where {R <: BlasFloat}
    copyto!(mat, fact.factors)
    LAPACK.potri!(fact.uplo, mat)
    copytri!(mat, fact.uplo, true)
    return mat
end

function chol_inv!(
    mat::Matrix{R},
    fact::Cholesky{R},
    ) where {R <: RealOrComplex{<:Real}}
    # this is how Julia computes the inverse, but it could be implemented better
    copyto!(mat, I)
    ldiv!(fact, mat)
    copytri!(mat, fact.uplo, true) # exactly symmetrize
    return mat
end


# helpers for updating symmetric/Hermitian eigendecomposition

update_eigen!(X::Matrix{<:BlasFloat}) = LAPACK.syev!('V', 'U', X)[1]

function update_eigen!(X::Matrix{<:RealOrComplex{<:Real}})
    F = eigen(Hermitian(X, :U))
    copyto!(X, F.vectors)
    return F.values
end


# helpers for symmetric outer product (upper triangle only)
# B = alpha * A' * A + beta * B

outer_prod!(
    A::Matrix{T},
    B::Matrix{T},
    alpha::Real,
    beta::Real,
    ) where {T <: LinearAlgebra.BlasReal} =
    BLAS.syrk!('U', 'T', alpha, A, beta, B)

outer_prod!(
    A::AbstractMatrix{Complex{T}},
    B::AbstractMatrix{Complex{T}},
    alpha::Real,
    beta::Real,
    ) where {T <: LinearAlgebra.BlasReal} =
    BLAS.herk!('U', 'C', alpha, A, beta, B)

outer_prod!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    alpha::Real,
    beta::Real,
    ) where {R <: RealOrComplex} =
    mul!(B, A', A, alpha, beta)


# ensure diagonal terms in symm/herm are not too small
function increase_diag!(A::Matrix{<:RealOrComplex{T}}) where {T <: Real}
    diag_pert = 1 + T(1e-5)
    diag_min = 10eps(T)
    @inbounds for j in 1:size(A, 1)
        A[j, j] = diag_pert * max(A[j, j], diag_min)
    end
    return A
end


# helpers for spectral outer products

function spectral_outer!(
    mat::AbstractMatrix{T},
    vecs::Union{Matrix{T}, Adjoint{T, Matrix{T}}},
    diag::AbstractVector{T},
    temp::Matrix{T},
    ) where {T <: Real}
    mul!(temp, vecs, Diagonal(diag))
    mul!(mat, temp, vecs')
    return mat
end

function spectral_outer!(
    mat::AbstractMatrix{T},
    vecs::Union{Matrix{T}, Adjoint{T, Matrix{T}}},
    symm::Symmetric{T},
    temp::Matrix{T},
    ) where {T <: Real}
    mul!(temp, vecs, symm)
    mul!(mat, temp, vecs')
    return mat
end


#=
nonsymmetric: LU
=#

abstract type DenseNonSymCache{T <: Real} end

mutable struct LAPACKNonSymCache{T <: BlasReal} <: DenseNonSymCache{T}
    copy_A
    AF
    ipiv
    info
    LAPACKNonSymCache{T}() where {T <: BlasReal} = new{T}()
end

function load_matrix(
    cache::LAPACKNonSymCache{T},
    A::Matrix{T};
    copy_A::Bool = true,
    ) where {T <: BlasReal}
    LinearAlgebra.require_one_based_indexing(A)
    LinearAlgebra.chkstride1(A)
    n = LinearAlgebra.checksquare(A)
    cache.copy_A = copy_A
    cache.AF = (copy_A ? zero(A) : A) # copy over A to new matrix or use A directly
    cache.ipiv = Vector{BlasInt}(undef, n)
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK functions
for (getrf, elty) in [(:dgetrf_, :Float64), (:sgetrf_, :Float32)]
    @eval begin
        function update_fact(cache::LAPACKNonSymCache{$elty}, A::AbstractMatrix{$elty})
            n = LinearAlgebra.checksquare(A)
            cache.copy_A && copyto!(cache.AF, A)

            # call dgetrf( m, n, a, lda, ipiv, info )
            ccall((@blasfunc($getrf), liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{BlasInt}),
                n, n, cache.AF, max(stride(cache.AF, 2), 1),
                cache.ipiv, cache.info)

            if cache.info[] < 0
                throw(ArgumentError("invalid argument #$(-cache.info[]) to LAPACK"))
            elseif 0 < cache.info[] <= n
                # @warn("factorization failed: #$(cache.info[])")
                return false
            elseif cache.info[] > n
                @warn("condition number is small: $(cache.rcond[])")
            end

            return true
        end
    end
end

for (getrs, elty) in [(:dgetrs_, :Float64), (:sgetrs_, :Float32)]
    @eval begin
        function inv_prod(cache::LAPACKNonSymCache{$elty}, X::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X)
            LinearAlgebra.chkstride1(X)

            # call dgetrs( trans, n, nrhs, a, lda, ipiv, b, ldb, info )
            ccall((@blasfunc($getrs), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                'N', size(cache.AF, 1), size(X, 2), cache.AF,
                max(stride(cache.AF, 2), 1), cache.ipiv, X, max(stride(X, 2), 1),
                cache.info)

            return X
        end
    end
end

mutable struct LUNonSymCache{T <: Real} <: DenseNonSymCache{T}
    copy_A
    AF
    fact
    LUNonSymCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(
    cache::LUNonSymCache{T},
    A::AbstractMatrix{T};
    copy_A::Bool = true,
    ) where {T <: Real}
    n = LinearAlgebra.checksquare(A)
    cache.copy_A = copy_A
    cache.AF = (copy_A ? zero(A) : A) # copy over A to new matrix or use A directly
    return cache
end

function update_fact(cache::LUNonSymCache{T}, A::AbstractMatrix{T}) where {T <: Real}
    cache.copy_A && copyto!(cache.AF, A)
    cache.fact = lu!(cache.AF, check = false)
    return issuccess(cache.fact)
end

inv_prod(cache::LUNonSymCache{T}, prod::AbstractVecOrMat{T}) where {T <: Real} =
    ldiv!(cache.fact, prod)

# default to LAPACKNonSymCache for BlasReals, otherwise generic LUNonSymCache
DenseNonSymCache{T}() where {T <: BlasReal} = LAPACKNonSymCache{T}()
DenseNonSymCache{T}() where {T <: Real} = LUNonSymCache{T}()

#=
symmetric indefinite: BunchKaufman (and LU fallback)
TODO add a generic BunchKaufman implementation to Julia and use that instead of LU for generic case
TODO try Aasen's version (http://www.netlib.org/lapack/lawnspdf/lawn294.pdf) and others
=#

abstract type DenseSymCache{T <: Real} end

mutable struct LAPACKSymCache{T <: BlasReal} <: DenseSymCache{T}
    copy_A
    AF
    ipiv
    work
    lwork
    info
    LAPACKSymCache{T}() where {T <: BlasReal} = new{T}()
end

function load_matrix(
    cache::LAPACKSymCache{T},
    A::Symmetric{T, <:AbstractMatrix{T}};
    copy_A::Bool = true,
    ) where {T <: BlasReal}
    LinearAlgebra.require_one_based_indexing(A.data)
    LinearAlgebra.chkstride1(A.data)
    n = LinearAlgebra.checksquare(A.data)
    cache.copy_A = copy_A
    cache.AF = (copy_A ? zero(A) : A) # copy over A to new matrix or use A directly
    cache.ipiv = Vector{BlasInt}(undef, n)
    cache.work = zeros(T, n) # this will be resized according to query
    cache.lwork = BlasInt(-1) # -1 initiates a query for optimal size of work
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK functions
for (sytrf_rook, elty) in [(:dsytrf_rook_, :Float64), (:ssytrf_rook_, :Float32)]
    @eval begin
        function update_fact(
            cache::LAPACKSymCache{$elty},
            A::Symmetric{$elty, <:AbstractMatrix{$elty}},
            )
            n = LinearAlgebra.checksquare(A)
            cache.copy_A && copyto!(cache.AF, A)
            AF = cache.AF.data

            # call dsytrf_rook( uplo, n, a, lda, ipiv, work, lwork, info )
            ccall((@blasfunc($sytrf_rook), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                cache.AF.uplo, n, AF, max(stride(AF, 2), 1),
                cache.ipiv, cache.work, cache.lwork, cache.info)

            if cache.lwork < 0 # query for optimal work size, and resize work before solving
                cache.lwork = BlasInt(real(cache.work[1]))
                resize!(cache.work, cache.lwork)
                return update_fact(cache, A)
            end

            if cache.info[] < 0
                throw(ArgumentError("invalid argument #$(-cache.info[]) to LAPACK"))
            elseif 0 < cache.info[] <= n
                # @warn("factorization failed: #$(cache.info[])")
                return false
            elseif cache.info[] > n
                @warn("condition number is small: $(cache.rcond[])")
            end

            return true
        end
    end
end

for (sytri_rook, elty) in [(:dsytri_rook_, :Float64), (:ssytri_rook_, :Float32)]
    @eval begin
        function invert(
            cache::LAPACKSymCache{$elty},
            X::Symmetric{$elty, Matrix{$elty}},
            )
            copyto!(X.data, cache.AF.data)

            # call dsytri_rook( uplo, n, a, lda, ipiv, work, info )
            ccall((@blasfunc($sytri_rook), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                X.uplo, size(X.data, 1), X.data, max(stride(X.data, 2), 1),
                cache.ipiv, cache.work, cache.info)

            return X
        end
    end
end

for (sytrs_rook, elty) in [(:dsytrs_rook_, :Float64), (:ssytrs_rook_, :Float32)]
    @eval begin
        function inv_prod(cache::LAPACKSymCache{$elty}, X::AbstractVecOrMat{$elty})
            LinearAlgebra.require_one_based_indexing(X)
            LinearAlgebra.chkstride1(X)
            AF = cache.AF.data

            # call dsytrs_rook( uplo, n, nrhs, a, lda, ipiv, b, ldb, info )
            ccall((@blasfunc($sytrs_rook), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                cache.AF.uplo, size(AF, 1), size(X, 2), AF,
                max(stride(AF, 2), 1), cache.ipiv, X, max(stride(X, 2), 1),
                cache.info)

            return X
        end
    end
end

mutable struct LUSymCache{T <: Real} <: DenseSymCache{T}
    copy_A
    AF
    fact
    LUSymCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(
    cache::LUSymCache{T},
    A::Symmetric{T, <:AbstractMatrix{T}};
    copy_A::Bool = true,
    ) where {T <: Real}
    n = size(A, 1)
    cache.copy_A = copy_A
    cache.AF = (copy_A ? zero(A) : A) # copy over A (symmetric) to new matrix or use A directly
    return cache
end

function update_fact(
    cache::LUSymCache{T},
    A::Symmetric{T, <:AbstractMatrix{T}},
    ) where {T <: Real}
    cache.copy_A && copyto!(cache.AF, A)
    cache.fact = lu!(cache.AF, check = false) # no generic symmetric indefinite factorization so fallback LU of symmetric matrix
    return issuccess(cache.fact)
end

function invert(
    cache::LUSymCache{T},
    X::Symmetric{T, <:AbstractMatrix{T}},
    ) where {T <: Real}
    copyto!(X.data, I)
    ldiv!(cache.fact, X.data) # just ldiv an identity matrix - LinearAlgebra currently does the same
    return X
end

inv_prod(cache::LUSymCache{T}, prod::AbstractVecOrMat{T}) where {T <: Real} =
    ldiv!(cache.fact, prod)

# default to LAPACKSymCache for BlasReals, otherwise generic LUSymCache
DenseSymCache{T}() where {T <: BlasReal} = LAPACKSymCache{T}()
DenseSymCache{T}() where {T <: Real} = LUSymCache{T}()

#=
symmetric positive definite: unpivoted Cholesky
=#

abstract type DensePosDefCache{T <: Real} end

mutable struct LAPACKPosDefCache{T <: BlasReal} <: DensePosDefCache{T}
    copy_A
    AF
    info
    LAPACKPosDefCache{T}() where {T <: BlasReal} = new{T}()
end

function load_matrix(
    cache::LAPACKPosDefCache{T},
    A::Symmetric{T, <:AbstractMatrix{T}};
    copy_A::Bool = true,
    ) where {T <: BlasReal}
    LinearAlgebra.require_one_based_indexing(A.data)
    LinearAlgebra.chkstride1(A.data)
    n = LinearAlgebra.checksquare(A.data)
    cache.copy_A = copy_A
    cache.AF = (copy_A ? zero(A) : A) # copy over A to new matrix or use A directly
    cache.info = Ref{BlasInt}()
    return cache
end

# wrap LAPACK functions
for (potrf, elty) in [(:dpotrf_, :Float64), (:spotrf_, :Float32)]
    @eval begin
        function update_fact(
            cache::LAPACKPosDefCache{$elty},
            A::Symmetric{$elty, <:AbstractMatrix{$elty}},
            )
            n = size(cache.AF, 1)
            cache.copy_A && copyto!(cache.AF, A)
            AF = cache.AF.data

            # call dpotrf( uplo, n, a, lda, info )
            ccall((@blasfunc($potrf), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                cache.AF.uplo, n, AF, max(stride(AF, 2), 1),
                cache.info)

            if cache.info[] < 0
                throw(ArgumentError("invalid argument #$(-cache.info[]) to LAPACK"))
            elseif 0 < cache.info[] <= n
                # @warn("factorization failed: #$(cache.info[])")
                return false
            elseif cache.info[] > n
                @warn("condition number is small: $(cache.rcond[])")
                return false
            end

            return true
        end
    end
end

for (potri, elty) in [(:dpotri_, :Float64), (:spotri_, :Float32)]
    @eval begin
        function invert(
            cache::LAPACKPosDefCache{$elty},
            X::Symmetric{$elty, Matrix{$elty}},
            )
            copyto!(X.data, cache.AF.data)

            # call dpotri( uplo, n, a, lda, info )
            ccall((@blasfunc($potri), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                X.uplo, size(X.data, 1), X.data, max(stride(X.data, 2), 1),
                cache.info)

            return X
        end
    end
end

for (potrs, elty) in [(:dpotrs_, :Float64), (:spotrs_, :Float32)]
    @eval begin
        function inv_prod(
            cache::LAPACKPosDefCache{$elty},
            X::AbstractVecOrMat{$elty},
            )
            LinearAlgebra.require_one_based_indexing(X)
            LinearAlgebra.chkstride1(X)
            AF = cache.AF.data

            # call dpotrs( uplo, n, nrhs, a, lda, b, ldb, info )
            ccall((@blasfunc($potrs), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}),
                cache.AF.uplo, size(AF, 1), size(X, 2), AF,
                max(stride(AF, 2), 1), X, max(stride(X, 2), 1), cache.info)

            return X
        end
    end
end

mutable struct CholPosDefCache{T <: Real} <: DensePosDefCache{T}
    copy_A
    AF
    fact
    CholPosDefCache{T}() where {T <: Real} = new{T}()
end

function load_matrix(
    cache::CholPosDefCache{T},
    A::Symmetric{T, <:AbstractMatrix{T}};
    copy_A::Bool = true,
    ) where {T <: Real}
    n = LinearAlgebra.checksquare(A)
    cache.copy_A = copy_A
    cache.AF = (copy_A ? zero(A) : A) # copy over A (symmetric) to new matrix or use A directly
    return cache
end

function update_fact(
    cache::CholPosDefCache{T},
    A::Symmetric{T, <:AbstractMatrix{T}},
    ) where {T <: Real}
    cache.copy_A && copyto!(cache.AF, A)
    cache.fact = cholesky!(cache.AF, check = false)
    if !issuccess(cache.fact) && cache.copy_A # fallback to LU of symmetric matrix
        copyto!(cache.AF, A)
        cache.fact = lu!(cache.AF, check = false)
    end
    return issuccess(cache.fact)
end

function invert(
    cache::CholPosDefCache{T},
    X::Symmetric{T, <:AbstractMatrix{T}},
    ) where {T <: Real}
    copyto!(X.data, I)
    ldiv!(cache.fact, X.data) # just ldiv an identity matrix - LinearAlgebra currently does the same
    return X
end

inv_prod(cache::CholPosDefCache{T}, prod::AbstractVecOrMat{T}) where {T <: Real} =
    ldiv!(cache.fact, prod)

sqrt_prod(
    cache::Union{LAPACKPosDefCache{T}, CholPosDefCache{T}},
    prod::AbstractVecOrMat{T},
    ) where {T <: Real} = lmul!(UpperTriangular(cache.AF.data), prod)

inv_sqrt_prod(
    cache::Union{LAPACKPosDefCache{T}, CholPosDefCache{T}},
    prod::AbstractVecOrMat{T},
    ) where {T <: Real} = ldiv!(UpperTriangular(cache.AF.data)', prod)

# default to LAPACKPosDefCache for BlasReals, otherwise generic CholPosDefCache
DensePosDefCache{T}() where {T <: BlasReal} = LAPACKPosDefCache{T}()
DensePosDefCache{T}() where {T <: Real} = CholPosDefCache{T}()
