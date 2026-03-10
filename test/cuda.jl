"""Test basic CUDA/CUSPARSE/CUSOLVER functionality on this package's test models
"""

import MadAI
import NLPModelsJuMP
import MadNLP
import LinearAlgebra
import SparseArrays
import Random
import CUDA
import CUDA.CUSPARSE: CuSparseMatrixCSR
using Test

include("models.jl")

"""
The pivot matrix is:

  | W G'|
  | G   |

This function assumes that G is already lower triangular
"""
function _get_pivot_lowertri_order(matrix)
    @assert matrix.n == matrix.m
    N = matrix.n
    ny = Int(N / 2)
    rp = zeros(Int, N)
    cp = zeros(Int, N)
    for i = 1:ny
        cp[i] = i
        cp[i+ny] = 2*ny-i+1
        rp[i] = i+ny
        rp[i+ny] = ny-i+1
    end
    return rp, cp
end

function test_cuda_linearsolve_synthetic()
    model, info = get_synthetic_nn_model()
    formulation = info.formulation

    # I will construct random RHSs below. Note that this must happen
    # _after_ constructing the synthetic model, which resets the seed.
    Random.seed!(101)

    pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
    pivot_indices = MadAI.get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    # We probably won't use these here... It is an open question whether or not
    # we can do something faster than CUSPARSE with our knowledge of where the
    # dense blocks are.
    #blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)

    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(nlp)
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    C = kkt_matrix[pivot_indices, pivot_indices]
    N = C.n
    nrhs = 100
    rhs = rand(N, nrhs)

    ny = Int(N / 2)

    # Lower triangle-to-full
    C = C + C' - LinearAlgebra.Diagonal(C)
    roworder, colorder = _get_pivot_lowertri_order(C)
    C_perm = C[roworder, colorder]
    # This relies on the pivot indices being provided in a lower triangular order.
    # Because I parse the NN formulation to obtain these indices, I know this is
    # the case. In general, I may have to reorder these pivot indices.
    @assert LinearAlgebra.istril(C_perm)

    C_gpu = CuSparseMatrixCSR(C_perm)
    LT_gpu = LinearAlgebra.LowerTriangular(C_gpu)
    rhs_gpu = CUDA.CuMatrix(rhs)
    sol_gpu = CUDA.CuMatrix(copy(rhs))
    LinearAlgebra.ldiv!(sol_gpu, LT_gpu, rhs_gpu)

    LT_cpu = LinearAlgebra.LowerTriangular(C_perm)
    sol_cpu = copy(rhs)
    LinearAlgebra.ldiv!(sol_cpu, LT_cpu, rhs)
    r_cpu = rhs - C_perm * sol_cpu
    Δ_cpu = LinearAlgebra.norm(r_cpu, Inf)
    println("CPU max residual: $Δ_cpu")
    @test Δ_cpu <= 1e-4

    # mul with LowerTriangular(C) raises a scalar indexing error...
    # I guess this wrapper is only efficient for the backsolve?
    #r_gpu = rhs_gpu - LT_gpu * sol_gpu
    r_gpu = rhs_gpu - C_gpu * sol_gpu
    Δ_gpu = LinearAlgebra.norm(r_gpu, Inf)
    println("GPU max residual: $Δ_gpu")
    @test Δ_gpu <= 1e-4
    return
end

function test_cuda_construct_schur_synthetic(; sparse = false)
    model, info = get_synthetic_nn_model()
    formulation = info.formulation

    pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
    pivot_indices = MadAI.get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)

    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(nlp)
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    C = kkt_matrix[pivot_indices, pivot_indices]
    N = kkt_matrix.n
    pivot_dim = C.n
    schur_dim = kkt_matrix.n - pivot_dim

    index_set = Set(pivot_indices)
    reduced_indices = filter(i -> !(i in index_set), 1:N)
    # I don't need the RHS to construct the Schur complement, but it might be nice
    # to test the full solve here, in which case I will need it.
    #orig_rhs_reduced = rhs[reduced_indices]
    #orig_rhs_pivot = rhs[pivot_indices]
    P = pivot_indices
    R = reduced_indices
    A = kkt_matrix[R, R]
    B = kkt_matrix[P, R] + kkt_matrix[R, P]'

    C = C + C' - LinearAlgebra.Diagonal(C)
    roworder, colorder = _get_pivot_lowertri_order(C)
    C_perm = C[roworder, colorder]
    # This relies on the pivot indices being provided in a lower triangular order.
    # Because I parse the NN formulation to obtain these indices, I know this is
    # the case. In general, I may have to reorder these pivot indices.
    @assert LinearAlgebra.istril(C_perm)

    if sparse
        A_gpu = CuSparseMatrixCSR(A)
        # Do I need to pre-allocate all nonzeros that can possibly be filled?
        S_gpu = CuSparseMatrixCSR(A)
    else
        A_gpu = CUDA.CuMatrix(A)
        S_gpu = CUDA.zeros(Float64, schur_dim, schur_dim)
    end
    B_gpu = CUDA.CuMatrix(B)
    C_gpu = CuSparseMatrixCSR(C_perm)
    LT_gpu = LinearAlgebra.LowerTriangular(C_gpu)
    # I could just override B...
    temp = CUDA.CuMatrix(copy(B_gpu))

    LinearAlgebra.ldiv!(temp, LT_gpu, B_gpu)
    S_gpu .= A_gpu - B_gpu' * temp
    return
end

@testset "basic-cuda" begin
    test_cuda_linearsolve_synthetic()
    test_cuda_construct_schur_synthetic()
end
