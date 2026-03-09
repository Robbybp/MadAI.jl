import MadAI
import MadNLP
import MadNLPHSL
import MathProgIncidence as MPIN
using Test
include("models.jl")

function test_solve_synthetic_model()
    model, info = get_synthetic_nn_model(; Activation = () -> MOAI.Tanh(), hidden_dim = 256)
    igraph = MPIN.IncidenceGraphInterface(model; include_inequality = false)
    println(igraph)
    madnlp = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "linear_solver" => MadNLPHSL.Ma57Solver,
        "tol" => 1e-6,
    )
    JuMP.set_optimizer(model, madnlp)
    JuMP.optimize!(model)
    @test JuMP.is_solved_and_feasible(model)
    return
end

@testset "solve-synthetic" begin
    test_solve_synthetic_model()
end
