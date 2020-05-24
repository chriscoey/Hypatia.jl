import Hypatia
import Hypatia.Cones
import Hypatia.Models
import Hypatia.Solvers
using Test

function test_step(T::Type)
    c = T[1, 1, 1]
    A = Diagonal(one(T) * I, 3)
    b = rand(T, 3)
    G = Diagonal(-one(T) * , 3)
    h = rand(T, 3)
    hypoperlog_cone = Cones.HypoPerLog{T}(3)
    cones = Cones.Cone[hypoperlog_cone]



    model = Models.Model{T}(c, A, b, G, h, cones)
    stepper = Solvers.Stepper{T}
    solver = Solvers.Solver{T}(stepper = stepper)
    SO.load(solver, new_model)
    step(solver)
end
