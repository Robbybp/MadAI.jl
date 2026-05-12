import LinearAlgebra
import MadAI
import MadNLP
import JuMP
import MathOptAI
import NLPModelsJuMP
import Random

#if !isdefined(@__MODULE__, :_HELPERS_INCLUDED)
#const _HELPERS_INCLUDED = true

function _helper_full_kkt(M)
    return M + M' - LinearAlgebra.Diagonal(M)
end

function _helper_pivot_indices(model, info::NamedTuple)
    if !haskey(info, :variables) || !haskey(info, :constraints)
        return nothing
    end
    indices = MadAI.get_kkt_indices(model, info.variables, info.constraints)
    return convert(Vector{Int32}, indices)
end

function _helper_solve_columns!(solver, solutions, rhs)
    for j in axes(rhs, 2)
        solution = copy(view(rhs, :, j))
        MadNLP.solve!(solver, solution)
        solutions[:, j] .= solution
    end
    return solutions
end

function _test_factorize_and_solve(
    solver,
    matrix;
    nrhs::Int = 10,
    rng::Random.AbstractRNG = Random.MersenneTwister(1),
)
    MadNLP.factorize!(solver)
    inertia = MadNLP.inertia(solver)
    rhs = rand(rng, size(matrix, 1), nrhs)
    solutions = similar(rhs)
    _helper_solve_columns!(solver, solutions, rhs)
    residuals = rhs - _helper_full_kkt(matrix) * solutions
    residual_norms = [
        LinearAlgebra.norm(view(residuals, :, j), Inf) for j in axes(residuals, 2)
    ]
    return (;
        inertia,
        residual_norms,
    )
end

"""
If we don't know anything about the linear solver, we won't
use the model's info NamedTuple to get pivot indices.
"""
function _test_solve_kkt(
    model::JuMP.Model,
    LinearSolver::Type{<:MadNLP.AbstractLinearSolver};
)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(nlp)
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    solver = LinearSolver(kkt_matrix)
    inertia, residual_norms = _test_factorize_and_solve(solver, kkt_matrix)
    return (;
        inertia,
        residual_norms,
    )
end

"""SchurComplementSolver. We extract the pivot indices then just use
the default pivot solver.
"""
function _test_solve_kkt(
    model::JuMP.Model,
    info::NamedTuple,
    LinearSolver::Type{MadAI.SchurComplementSolver};
)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(nlp)
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    pivot_indices = MadAI.get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = MadAI.SchurComplementOptions(; pivot_indices)
    solver = LinearSolver(kkt_matrix; opt)
    inertia, residual_norms = _test_factorize_and_solve(solver, kkt_matrix)
    return (;
        inertia,
        residual_norms,
    )
end

"""SchurComplementSolver with PivotSolver::BlockTriangularSolver.
`formulation` is not provided, so we don't use it to infer the blocks.
"""
function _test_solve_kkt(
    model::JuMP.Model,
    info::NamedTuple,
    LinearSolver::Type{MadAI.SchurComplementSolver},
    PivotSolver::Type{MadAI.BlockTriangularSolver},
)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(nlp)
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    pivot_indices = MadAI.get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = MadAI.SchurComplementOptions(; pivot_indices, PivotSolver)
    solver = LinearSolver(kkt_matrix; opt)
    inertia, residual_norms = _test_factorize_and_solve(solver, kkt_matrix)
    return (;
        inertia,
        residual_norms,
    )
end

"""SchurComplementSolver with PivotSolver::BlockTriangularSolver.
`formulation` _is_ provided, so we use it to infer the blocks
"""
function _test_solve_kkt(
    model::JuMP.Model,
    info::NamedTuple,
    formulation::MathOptAI.AbstractFormulation,
    LinearSolver::Type{MadAI.SchurComplementSolver},
    PivotSolver::Type{MadAI.BlockTriangularSolver},
)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(nlp)
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    pivot_indices = MadAI.get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
    opt = MadAI.SchurComplementOptions(; pivot_indices, PivotSolver, pivot_solver_opt)
    solver = LinearSolver(kkt_matrix; opt)
    inertia, residual_norms = _test_factorize_and_solve(solver, kkt_matrix)
    return (;
        inertia,
        residual_norms,
    )
end

#end
