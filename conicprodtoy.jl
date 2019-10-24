

function conic_prod!(c, a, b)
    c[1] = dot(a, b)
    @. @views c[2:end] = a[1] * b[2:end] + b[1] * a[2:end]
    return c
end

function conic_div!(c, a, b)
    n = length(a)
    mat = zeros(n, n)
    mat[1, 2:end] .= a[2:end]
    for i in 1:n
        mat[i, i] = a[1]
    end
    c .= Symmetric(mat, :U) \ b
    return c
end

n = 2
a = randn(n)
c = similar(a)
e1 = zeros(n)
e1[1] = 1
tol = 100eps()
#
# a_inv = conic_div!(c, a, e1)
# tmp = similar(c)
# @test conic_prod!(tmp, a_inv, a) ≈ e1 atol=tol rtol=tol # ok
#
# a = e1 + randn(n) * 1e-6 # looks realistic
# a_inv = conic_div!(c, a, e1)
# tmp = similar(c)
# @test conic_prod!(tmp, a_inv, a) ≈ e1 atol=tol rtol=tol # ok

using Hypatia
const CO = Hypatia.Cones

Random.seed!(1)

point = Vector(undef, n)
dual_point = Vector(undef, n)
cone = CO.EpiNormEucl{Float64}(n)
CO.setup_data(cone)
CO.reset_data(cone)

CO.set_initial_point(point, cone)
CO.set_initial_point(dual_point, cone)
point .+= 0.2 * (rand(n) .- 0.5) / 10000
dual_point .+= 0.2 * (rand(n) .- 0.5) / 10000
CO.load_point(cone, point)
CO.load_dual_point(cone, dual_point)
@test CO.is_feas(cone)
l = similar(cone.point)
CO.scalmat_prod!(l, cone.dual_point, cone)
l_inv = CO.scalvec_ldiv!(c, cone, e1)
@show l_inv
tmp = similar(point)
@test CO.conic_prod!(tmp, cone, l_inv, l) ≈ e1 atol=tol rtol=tol # ok
@show  CO.conic_prod!(tmp, cone, l_inv, l)
