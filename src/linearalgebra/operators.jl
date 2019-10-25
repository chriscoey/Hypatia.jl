#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

=#
using LinearAlgebra

# function jordan_ldiv(C::AbstractVecOrMat, A::Vector, B::AbstractVecOrMat)
#     m = length(A)
#     @assert m == size(B, 1)
#     @assert size(B) == size(C)
#     A1 = A[1]
#     A2m = view(A, 2:m)
#     schur = abs2(A1) - sum(abs2, A2m)
#     @views begin
#         copyto!(C[1, :], B[1, :] .* A1)
#         # C[1, :] .-= B[2:end, :]' * A2m
#         mul!(C[1, :], B[2:end, :]', -A2m, true, true)
#         C[2:end, :] = -A2m .* B[1, :]' .+ A2m * A2m' * B[2:end, :] / A1
#         C ./= schur
#         @. C[2:end, :] += B[2:end, :] / A1
#     end
#     return C
# end

function jordan_ldiv(C::AbstractVecOrMat, A::Vector, B::AbstractVecOrMat)
    m = length(A)
    @assert m == size(B, 1)
    @assert size(B) == size(C)
    A1 = A[1]
    A2m = view(A, 2:m)
    schur = abs2(A1) - sum(abs2, A2m)
    @views begin
        mul!(C[1, :], B[2:end, :]', A2m)
        @. C[2:end, :] = A2m * C[1, :]' / A1
        axpby!(A1, B[1, :], -1.0, C[1, :])
        @. C[2:end, :] -= A2m * B[1, :]'
        C ./= schur
        @. C[2:end, :] += B[2:end, :] / A1
    end
    return C
end

tol = 1e-14

n = 50
A = randn(n)
B = randn(n)
C = zeros(n)
jordan_ldiv(C, A, B)

B2 = copy(B)
B2[1] = dot(A, C)
@. @views B2[2:end] = A[1] * C[2:end] + C[1] * A[2:end]

@test B ≈ B2 atol=tol rtol=tol

m = 5
n = 10
A = randn(m)
A_arr = zeros(m, m)
A_arr[diagind(A_arr)] .= A[1]
A_arr[1, 2:end] .= A[2:end]
B = randn(m, n)
@test jordan_ldiv(zeros(m, n), A, B) ≈ Symmetric(A_arr, :U) \ B atol=tol rtol=tol
