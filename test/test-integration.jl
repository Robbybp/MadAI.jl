import JuMP
import NLPModelsJuMP
import MathOptAI as MOAI
import MadNLP
import MadNLPHSL
using Test

include("linalg.jl")
include("btsolver.jl")
include("nlpmodels.jl")
include("kkt-partition.jl")
include("models.jl")

function test_nn_kkt_solve()
    model, info = get_deterministic_nn_model()
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = partition_indices_by_layer(model, info.formulation; indices = pivot_indices)
    pivot_solver_opt = BlockTriangularOptions(; blocks)
    madnlp_options = Dict{Symbol,Any}(
        :ReducedSolver => MadNLPHSL.Ma57Solver,
        :PivotSolver => BlockTriangularSolver,
        :pivot_indices => pivot_indices,
        :pivot_solver_opt => pivot_solver_opt,
    )

    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(
        nlp;
        tol = 1e-6,
        print_level = MadNLP.INFO,
        max_iter = 0,
        linear_solver = SchurComplementSolver,
        madnlp_options...,
    )
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)

    ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
    opt = SchurComplementOptions(;
        pivot_indices,
        PivotSolver = BlockTriangularSolver,
        pivot_solver_opt,
    )
    schur = SchurComplementSolver(kkt_matrix; opt)

    MadNLP.factorize!(ma27)
    MadNLP.factorize!(schur)

    rhs = ones(kkt_matrix.m)
    sol_ma27 = copy(rhs)
    sol_schur = copy(rhs)
    MadNLP.solve!(ma27, sol_ma27)
    MadNLP.solve!(schur, sol_schur)
    #diff = sol_ma27 .- sol_schur
    #error = maximum(abs.(diff))
    #println("Error = $error")

    @test MadNLP.inertia(ma27) == MadNLP.inertia(schur)
    @test all(isapprox.(sol_ma27, sol_schur; atol=1e-8))
end

function test_madnlp_3iter()
    model, info = get_deterministic_nn_model()
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = partition_indices_by_layer(model, info.formulation; indices = pivot_indices)
    pivot_solver_opt = BlockTriangularOptions(; blocks)
    optimizer = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "linear_solver" => SchurComplementSolver,
        "tol" => 1e-6,
        "max_iter" => 3,
        "print_level" => MadNLP.INFO,
        "ReducedSolver" => MadNLPHSL.Ma57Solver,
        "PivotSolver" => BlockTriangularSolver,
        "pivot_indices" => pivot_indices,
        "pivot_solver_opt" => pivot_solver_opt,
    )
    JuMP.set_optimizer(model, optimizer)
    JuMP.optimize!(model)

    @test JuMP.termination_status(model) in (JuMP.ITERATION_LIMIT, JuMP.LOCALLY_SOLVED)
end

function _baseline_solve()
    model, _ = get_deterministic_nn_model()
    madnlp = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "tol" => 1e-6,
        "max_iter" => 50,
        "print_level" => MadNLP.INFO,
    )
    JuMP.set_optimizer(model, madnlp)
    JuMP.optimize!(model)
    return JuMP.value.(JuMP.all_variables(model))
end

function test_madnlp_solve()
    model, info = get_deterministic_nn_model()
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = partition_indices_by_layer(model, info.formulation; indices = pivot_indices)
    pivot_solver_opt = BlockTriangularOptions(; blocks)
    optimizer = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "linear_solver" => SchurComplementSolver,
        "pivot_indices" => pivot_indices,
        "tol" => 1e-6,
        "max_iter" => 50,
        "print_level" => MadNLP.INFO,
        "ReducedSolver" => MadNLPHSL.Ma57Solver,
        "PivotSolver" => BlockTriangularSolver,
        "pivot_indices" => pivot_indices,
        "pivot_solver_opt" => pivot_solver_opt,
    )
    JuMP.set_optimizer(model, optimizer)
    JuMP.optimize!(model)
    JuMP.assert_is_solved_and_feasible(model)
    solution = JuMP.value.(JuMP.all_variables(model))
    baseline_solution = _baseline_solve()
    diff = solution .- baseline_solution
    error = maximum(abs.(diff))
    println("Error = $error")
    @test JuMP.termination_status(model) == JuMP.LOCALLY_SOLVED
    @test all(isapprox.(solution, baseline_solution; atol=1e-8))
end

@testset "integration" begin
    test_nn_kkt_solve()
    test_madnlp_3iter()
    test_madnlp_solve()
end
