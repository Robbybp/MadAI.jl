"""Tests for Schur complement algorithm(s) on GPU
"""

import MadAI

import MathOptAI as MOAI
import JuMP
import MadNLP, MadNLPHSL
import MathOptInterface as MOI
import MathProgIncidence as MPIN
import NLPModels, NLPModelsJuMP
import Random
import SparseArrays

using Test

include("models.jl")
include("helpers.jl")

# This is basically the same code as we use for SchurComplementSolver...
function _test_solve_kkt(
    model::JuMP.Model,
    info::NamedTuple,
    LinearSolver::Type{MadAI.GpuSchurComplementSolver};
)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(nlp)
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    pivot_indices = MadAI.get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = MadAI.GpuSchurComplementOptions(; pivot_indices)
    solver = LinearSolver(kkt_matrix; opt)
    inertia, residual_norms = _test_factorize_and_solve(solver, kkt_matrix)
    return (;
        inertia,
        residual_norms,
    )
end

function test_tiny_schur_gpu_dense()
    model, info = make_tiny_model()
    results = _test_solve_kkt(
        model,
        info,
        MadAI.GpuSchurComplementSolver,
    )
    @test results.inertia == (6, 0, 5)
    @test all(results.residual_norms .<= 1e-8)
    return
end

@testset "GPU Schur complement" begin
    test_tiny_schur_gpu_dense()
end
