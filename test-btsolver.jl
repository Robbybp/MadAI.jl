import JuMP
import LinearAlgebra
import MadNLP
import MadNLPHSL
import MathProgIncidence
import NLPModels
import SparseArrays
using Test
using Printf

include("adversarial-image.jl")
include("btsolver.jl")
include("models.jl")
include("nlpmodels.jl")
include("linalg.jl")

function _test_matrix(
    csc::SparseArrays.SparseMatrixCSC;
    blocks = nothing,
    atol=1e-8,
    nrhs = 10,
    baseline_solver = "umfpack",
    skiptest = false,
    btsolver = nothing,
    symmetric = true,
)
    dim = csc.m
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)

    _t = time()
    if btsolver === nothing
        opt = BlockTriangularOptions(; blocks, symmetric)
        btsolver = BlockTriangularSolver(csc; opt)
    end
    t_init = time() - _t
    _t = time()
    MadNLP.factorize!(btsolver)
    t_fact = time() - _t
    sol = copy(rhs)
    _t = time()
    MadNLP.solve!(btsolver, sol)
    t_solve = time() - _t

    if !skiptest
        if baseline_solver == "umfpack"
            baseline = csc \ rhs
        elseif baseline_solver == "ma57"
            solver = MadNLPHSL.Ma57Solver(SparseArrays.tril(csc))
            MadNLP.factorize!(solver)
            baseline = copy(rhs)
            MadNLP.solve!(solver, baseline)
        else
            error("baseline_solver argument must be \"umfpack\" or \"ma57\"")
        end
        @test all(isapprox.(sol, baseline; atol))
        if !(all(isapprox.(sol, baseline; atol)))
            diff = abs.(sol .- baseline)
            ndiff = count(diff[:, 1] .> atol)
            maxdiff = maximum(diff)
            println("Solution does not match baseline")
            println("Max error: $maxdiff")
            println("N. errors: $ndiff / $dim")
        end
    end
    return (;
        time = (;
            initialize = t_init,
            factorize = t_fact,
            solve = t_solve,
        ),
        btsolver,
    )
end

function test_3x3_lt()
    matrix = [
        1.0 0.0 0.0;
        5.0 1.0 0.0;
        0.0 2.0 1.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    _test_matrix(matrix; symmetric = false)
    blocks = [([1,2], [1,2]), ([3], [3])]
    _test_matrix(matrix; blocks, symmetric = false)
    return
end

"""
This test exercises the case where an unsymmetric column permutation matrix is used.
(I.e., if PAQ is our BT matrix, Q != Q^T.) This is important as, for symmetric
column permutations, we can reorder columns with the forward or inverse permutation
matrix without changing the result.
"""
function test_3x3_lt_unsym_perm()
    matrix = [
        0.0 0.0 1.0;
        1.0 0.0 5.0;
        2.0 1.0 0.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    _test_matrix(matrix; symmetric = false)
    blocks = [([1], [3]), ([3,2], [2,1])]
    _test_matrix(matrix; blocks, symmetric = false)
    return
end

function test_4x4_blt()
    matrix = [
        0.0 1.0 0.0 0.0;
        1.0 5.0 0.0 3.0;
        0.0 0.0 4.0 2.0;
        2.0 0.0 0.0 2.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    _test_matrix(matrix; symmetric = false)
    return
end

function test_5x5_large_block()
    # This matrix has a 4x4 block and a 1x1 block
    matrix = [
        1.0 0.0 0.0 2.0 0.0;
        3.0 1.0 2.0 0.0 0.0;
        1.0 0.0 1.0 1.0 0.0;
        2.0 3.0 2.0 1.0 0.0;
        0.0 4.0 0.0 0.0 1.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    _test_matrix(matrix; symmetric = false)
    return
end

function test_nn_jacobian()
    # NOTE: This model is constructed with random numbers.
    # TODO: Use a deterministic model here.
    model, info = make_small_nn_model()
    layers = get_layers(info.formulation)
    var_con_by_layer = [get_vars_cons(l) for l in layers]
    # ... do FixRef constraints not get picked up by NLPModels?
    #JuMP.fix.(model[:x], 2.0; force = true)
    input_cons = JuMP.@constraint(model, model[:x] .== 2.0)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    vars, cons = get_var_con_order(model)
    var_index_map = Dict((var, i) for (i, var) in enumerate(vars))
    con_index_map = Dict((con, i) for (i, con) in enumerate(cons))
    input_block = [
        ([con_index_map[c] for c in input_cons], [var_index_map[v] for v in model[:x]])
    ]
    output_blocks = [
        (
            [con_index_map[c] for c in cons],
            [var_index_map[v] for v in vars],
        )
        for (vars, cons) in var_con_by_layer
    ]
    blocks = vcat(input_block, output_blocks)
    display(blocks)
    x = [something(JuMP.start_value(var), 1.0) for var in vars]
    matrix = NLPModels.jac(nlp, x)
    display(matrix)
    info = _test_matrix(matrix; blocks, symmetric = false)

    # Test a re-solve with new values
    x2 = [2.0 for var in vars]
    jac2 = NLPModels.jac(nlp, x2)
    matrix.nzval .= jac2.nzval
    btsolver = info.btsolver
    _test_matrix(matrix; blocks, btsolver, symmetric = false)
    return
end

function test_nn_kkt()
    model, info = make_small_nn_model()
    nlp, kkt_system, kkt_matrix = get_kkt(model)
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    kkt_matrix = fill_upper_triangle(kkt_matrix)
    pivot_dim = length(pivot_indices)
    pivot_matrix = kkt_matrix[pivot_indices, pivot_indices]

    # Filter out constraint regularization nonzeros
    # By convention, constraints are the second half of the pivot indices
    to_ignore = Set(Int(pivot_dim / 2 + 1):pivot_dim)
    I, J, V = SparseArrays.findnz(pivot_matrix)
    to_retain = filter(k -> !(I[k] in to_ignore && J[k] in to_ignore), 1:length(I))
    I = I[to_retain]
    J = J[to_retain]
    V = V[to_retain]
    pivot_matrix = SparseArrays.sparse(I, J, V, pivot_dim, pivot_dim)

    _test_matrix(pivot_matrix)
end

function test_nn_kkt_symmetric_inverse()
    model, info = make_small_nn_model()
    nlp, kkt_system, kkt_matrix = get_kkt(model)
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    kkt_matrix = fill_upper_triangle(kkt_matrix)
    pivot_dim = length(pivot_indices)
    pivot_matrix = kkt_matrix[pivot_indices, pivot_indices]

    # Filter out constraint regularization nonzeros
    # By convention, constraints are the second half of the pivot indices
    to_ignore = Set(Int(pivot_dim / 2 + 1):pivot_dim)
    I, J, V = SparseArrays.findnz(pivot_matrix)
    to_retain = filter(k -> !(I[k] in to_ignore && J[k] in to_ignore), 1:length(I))
    I = I[to_retain]
    J = J[to_retain]
    V = V[to_retain]
    pivot_matrix = SparseArrays.sparse(I, J, V, pivot_dim, pivot_dim)

    # By using the identity matrix as the RHS, we recover the inverse.
    rhs = LinearAlgebra.diagm(ones(pivot_dim))
    btsolver = BlockTriangularSolver(pivot_matrix)
    MadNLP.factorize!(btsolver)
    sol = copy(rhs)
    MadNLP.solve!(btsolver, sol)
    baseline = pivot_matrix \ rhs
    @test all(isapprox.(sol, baseline; atol = 1e-8))

    symmetric_diffs = []
    for i in 1:pivot_dim
        for j in 1:(i-1)
            push!(symmetric_diffs, abs(sol[i,j] - sol[j, i]))
        end
    end
    @test all(symmetric_diffs .<= 1e-8)
end

@testset "block-triangular" begin
    test_3x3_lt()
    test_3x3_lt_unsym_perm()
    test_4x4_blt()
    test_5x5_large_block()
    test_nn_jacobian()
    test_nn_kkt()
    test_nn_kkt_symmetric_inverse()
end
