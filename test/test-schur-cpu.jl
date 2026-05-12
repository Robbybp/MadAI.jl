"""Tests for the Schur complement algorithm
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

"""Tiny model, Schur complement solver with default subsolvers"""
function test_tiny_schur_default()
    model, info = make_tiny_model()
    results = _test_solve_kkt(
        model,
        info,
        MadAI.SchurComplementSolver,
    )
    @test results.inertia == (6, 0, 5)
    @test all(results.residual_norms .<= 1e-8)
    return
end

"""Tiny model, Schur complement with BT subsolver"""
function test_tiny_schur_bt()
    model, info = make_tiny_model()
    results = _test_solve_kkt(
        model,
        info,
        MadAI.SchurComplementSolver,
        MadAI.BlockTriangularSolver,
    )
    @test results.inertia == (6, 0, 5)
    @test all(results.residual_norms .<= 1e-8)
    return
end

"""Synthetic NN model with default subsolver

We are very inaccurate on the synthetic NN model, for some reason.
MA57 (no Schur complement) is fine...
"""
#function test_synthetic_schur_default()
#    model, info = get_synthetic_nn_model()
#    # According
#    ma57_results = run_initial_kkt_solver_test(model, MadNLPHSL.Ma57Solver)
#    results = run_initial_kkt_solver_test(
#        model,
#        info,
#        MadAI.SchurComplementSolver,
#    )
#    println(ma57_results)
#    println(results)
#    @test results.inertia == (1729, 0, 1601)
#end
function test_synthetic_schur_BT()
    model, info = get_synthetic_nn_model()
    ma57_results = _test_solve_kkt(model, MadNLPHSL.Ma57Solver)
    results = _test_solve_kkt(
        model,
        info,
        info.formulation,
        MadAI.SchurComplementSolver,
        MadAI.BlockTriangularSolver,
    )
    println(ma57_results)
    println(results)
    @test results.inertia == (1729, 0, 1601)
    return
end

"""Deterministic NN model with default subsolver

This NN has fine accuracy, for some reason...
"""
function test_deterministic_NN_schur_default()
    model, info = get_deterministic_nn_model()
    results = _test_solve_kkt(
        model,
        info,
        MadAI.SchurComplementSolver,
    )
    @test results.inertia == (520, 0, 420)
    @test all(results.residual_norms .<= 1e-7)
    return
end

function test_deterministic_NN_schur_BT()
    model, info = get_deterministic_nn_model()
    results = _test_solve_kkt(
        model,
        info,
        info.formulation,
        MadAI.SchurComplementSolver,
        MadAI.BlockTriangularSolver,
    )
    @test results.inertia == (520, 0, 420)
    @test all(results.residual_norms .<= 1e-7)
    return
end

function test_solve_tiny_with_jump()
    model, info = make_tiny_model()
    _test_solve_with_jump(model, info, MadAI.SchurComplementSolver)
    @test JuMP.termination_status(model) == JuMP.LOCALLY_SOLVED
    return
end

function test_solve_tiny_with_jump_BT()
    model, info = make_tiny_model()
    _test_solve_with_jump(
        model,
        info,
        MadAI.SchurComplementSolver,
        MadAI.BlockTriangularSolver,
    )
    @test JuMP.termination_status(model) == JuMP.LOCALLY_SOLVED
    return
end

function test_solve_deterministic_NN_with_jump()
    model, info = get_deterministic_nn_model()
    _test_solve_with_jump(model, info, MadAI.SchurComplementSolver)
    @test JuMP.termination_status(model) == JuMP.LOCALLY_SOLVED
    return
end

function test_solve_deterministic_NN_with_jump_BT()
    model, info = get_deterministic_nn_model()
    _test_solve_with_jump(
        model,
        info,
        info.formulation,
        MadAI.SchurComplementSolver,
        MadAI.BlockTriangularSolver,
    )
    @test JuMP.termination_status(model) == JuMP.LOCALLY_SOLVED
    return
end

@testset "CPU Schur complement" begin
    test_tiny_schur_default()
    test_tiny_schur_bt()
    test_solve_tiny_with_jump()
    test_solve_tiny_with_jump_BT()

    #test_synthetic_schur_default()
    #test_synthetic_schur_BT()
    test_deterministic_NN_schur_default()
    test_deterministic_NN_schur_BT()
    test_solve_deterministic_NN_with_jump()
    test_solve_deterministic_NN_with_jump_BT()
end
