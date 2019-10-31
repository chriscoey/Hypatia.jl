
using ForwardDiff
using LinearAlgebra

function bar(point)
    u = point[1]
    v = point[2]
    w = point[3]
    return -log(v * log(u / v) - w) - log(u) - log(v)
end


u = rand()
v = rand()
w = v * log(u / v) - 1
x = [u, v, w]

fx = bar(x)
g = ForwardDiff.gradient(bar, x)
H = ForwardDiff.hessian(bar, x)
d3 = ForwardDiff.jacobian(x -> ForwardDiff.hessian(bar, x), x)

# check log-homog property that F'''(x)[x] = -2F''(x)
d3x = d3 * x
d3xt = reshape(d3x, 3, 3)
@assert d3xt ≈ -2 * H

# test
Da_s = rand(3)
Da_z = rand(3)

FD_Hinv_Da_z = Symmetric(H) \ Da_z
FD_corr = reshape(d3 * Da_s, 3, 3) * FD_Hinv_Da_z / -2
