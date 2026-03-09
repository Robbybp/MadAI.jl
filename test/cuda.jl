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

function test_cuda_synthetic()
    model, info = get_synthetic_nn_model()

    # I will construct random RHSs below. Note that this must happen
    # _after_ constructing the synthetic model, which resets the seed.
    Random.seed!(101)

    formulation = info.formulation

    pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
    pivot_indices = MadAI.get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    # We probably won't use these here...
    blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)

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

    # 1. Reorder matrix to be lower triangular
    # 2. Backsolve with CUSOLVER

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

@testset "basic-cuda" begin
    test_cuda_synthetic()
end
