using LinearAlgebra
using SparseArrays

U = 3
n = U - 1

GI = vcat(fill(1, n), collect(2:U))
GJ = vcat(collect(1:n), collect(1:n))
GV = vcat(fill(1, n), fill(-1, n))
G = sparse(GI, GJ, GV, U, n)

H = Symmetric([1 0 0; 2 3 0; 4 5 6], :L)

A = Matrix(H * G)
B = G' * A
