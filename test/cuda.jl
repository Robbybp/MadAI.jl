"""Test basic CUDA/CUSPARSE/CUSOLVER functionality on this package's test models
"""

import MadAI
import NLPModelsJuMP
import MadNLP
import MadNLPHSL
import LinearAlgebra
import SparseArrays
import Random
import CUDA
import CUDA.CUSPARSE: CuSparseMatrixCSR
import CUDA.CUSOLVER
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
    # Transfer B to GPU. We don't permute it yet because we still need the original
    B_gpu = CUDA.CuMatrix(B)
    # Allocate a permuted B on GPU
    B_gpu_perm = B_gpu[roworder, :]
    C_gpu = CuSparseMatrixCSR(C_perm)
    LT_gpu = LinearAlgebra.LowerTriangular(C_gpu)
    # This will store the intermediate product C^-1 B
    temp = CUDA.CuMatrix(copy(B_gpu_perm))

    # Compute C^-1 B (in the permuted-column space)
    LinearAlgebra.ldiv!(temp, LT_gpu, B_gpu_perm)
    # TODO: This can be done with less intermediate memory usage
    # We use the original B. The intermediate product's rows have
    # the inverse column permutation (of C) applied. So we apply the
    # forward column permutation to these rows.
    BTCB_gpu = B_gpu' * temp[invperm(colorder), :]
    S_gpu .= A_gpu - BTCB_gpu

    if sparse
        # FIXME: This is buggy. My construction of S_gpu mixes sparse and dense
        # types, which causes problems
        S_cpu = SparseArrays.SparseMatrixCSC(S_gpu)
        BTCB_cpu = SparseArrays.SparseMatrixCSC(BTCB_gpu)
    else
        S_cpu = Matrix(S_gpu)
        BTCB_cpu = Matrix(BTCB_gpu)
    end

    # Since A is only the lower triangle, S isn't symmetric, so we check
    # the symmetry of (B^T C^-1 B)
    is_sym = LinearAlgebra.issymmetric(BTCB_cpu)
    println("BTCB symmetric: $is_sym")
    sym_error = abs.(BTCB_cpu - BTCB_cpu')
    println("Max(|S - S'|) = $(maximum(sym_error))")

    # Remove extra nonzeros in the upper triangle (from BTCB)
    S_gpu .= LinearAlgebra.tril(S_gpu)

    return

    # Tests for the Schur complement:
    # - symmetric (or close to it)
    # - nonsingular
    # - Can be used to solve the original linear system
    # - Yields correct inertia

    nrhs = 5
    rhs_cpu = rand(N, nrhs)
    ma57 = MadNLPHSL.Ma57Solver(kkt_matrix)
    MadNLP.factorize!(ma57)
    sol_cpu = copy(rhs_cpu)
    MadNLP.solve!(ma57, sol_cpu)

    if !sparse
        rhs_reduced = rhs_cpu[R, :]
        rhs_pivot = rhs_cpu[P, :]
        rhs_pivot_perm = rhs_pivot[roworder, :]

        rhs_reduced_gpu = CUDA.CuMatrix(rhs_reduced)
        rhs_pivot_perm_gpu = CUDA.CuMatrix(rhs_pivot_perm)

        # Solve C * Z = B (already computed in temp) and C * y = g
        Cg_gpu = CUDA.CuMatrix(copy(rhs_pivot_perm_gpu))
        LinearAlgebra.ldiv!(Cg_gpu, LT_gpu, rhs_pivot_perm_gpu)

        schur_rhs_gpu = rhs_reduced_gpu - B_gpu' * Cg_gpu
        x_gpu = CUDA.CuMatrix(copy(schur_rhs_gpu))

        # LBL'
        F_gpu, ipiv_gpu, _ = CUSOLVER.sytrf!('L', S_gpu)
        ipiv_gpu = CUDA.CuVector{Int64}(ipiv_gpu)
        CUSOLVER.sytrs!('L', F_gpu, ipiv_gpu, x_gpu)

        # LU
        #F_gpu, ipiv_gpu, _ = CUSOLVER.Xgetrf!(S_gpu)
        ## 'N' for "not transpose", of course
        #CUSOLVER.Xgetrs!('N', F_gpu, ipiv_gpu, x_gpu)

        rhs_pivot_corr_gpu = rhs_pivot_perm_gpu - B_gpu * x_gpu
        y_perm_gpu = CUDA.CuMatrix(copy(rhs_pivot_corr_gpu))
        LinearAlgebra.ldiv!(y_perm_gpu, LT_gpu, rhs_pivot_corr_gpu)

        x = Matrix(x_gpu)
        y_perm = Matrix(y_perm_gpu)
        y = y_perm[invperm(roworder), :]

        sol_schur = zeros(N, nrhs)
        sol_schur[R, :] = x
        sol_schur[P, :] = y

        err = maximum(abs.(sol_schur - sol_cpu))
        println("Schur GPU vs MA57 CPU max error: $err")
        @test err <= 1e-4
    end
    return
end

function test_cpu_construct_schur_synthetic(; use_hsl = false)
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

    index_set = Set(pivot_indices)
    reduced_indices = filter(i -> !(i in index_set), 1:N)
    P = pivot_indices
    R = reduced_indices
    A = kkt_matrix[R, R]
    B = kkt_matrix[P, R] + kkt_matrix[R, P]'

    if use_hsl
        # This alternative implementation yields no error. This seems to imply
        # that the error is coming from the LowerTriangular backsolve
        pivot_solver = MadNLPHSL.Ma57Solver(C)
        # TODO: HSL must have a triangular solve method I can use...
        MadNLP.factorize!(pivot_solver)
        pivot_inertia = MadNLP.inertia(pivot_solver)
        println("Pivot (C) inertia: $pivot_inertia")
        temp = copy(B)
        MadNLP.solve!(pivot_solver, temp)
        BTCB = B' * temp
    else
        # Creating the full pivot matrix and permuted B is only necessary
        # when we're solving with a triangular method
        C = C + C' - LinearAlgebra.Diagonal(C)
        @assert LinearAlgebra.issymmetric(C)
        roworder, colorder = _get_pivot_lowertri_order(C)
        C_perm = C[roworder, colorder]
        @assert LinearAlgebra.istril(C_perm)
        B_perm = B[roworder, :]
        temp = LinearAlgebra.LowerTriangular(C_perm) \ B_perm
        temp_unperm = temp[invperm(colorder), :]
        BTCB = B' * temp_unperm
    end
    S = A - SparseArrays.sparse(LinearAlgebra.tril(BTCB))

    is_sym = LinearAlgebra.issymmetric(BTCB)
    println("CPU Schur symmetric: $is_sym")
    sym_error = abs.(BTCB - BTCB')
    println("CPU max(|S - S'|) = $(maximum(sym_error))")
    avg_sym_error = sum(sym_error) / length(sym_error)
    println("Avg CPU sym error = $(avg_sym_error)")

    nrhs = 5
    rhs = rand(N, nrhs)
    rhs_reduced = rhs[R, :]
    rhs_pivot = rhs[P, :]

    if use_hsl
        Cinv_rhs_pivot = copy(rhs_pivot)
        MadNLP.solve!(pivot_solver, Cinv_rhs_pivot)
    else
        rhs_pivot_perm = rhs_pivot[roworder, :]
        Cinv_rhs_pivot = LinearAlgebra.LowerTriangular(C_perm) \ rhs_pivot_perm
        Cinv_rhs_pivot = Cinv_rhs_pivot[invperm(colorder), :]
    end
    # Backsolving through the Schur complement is the same no matter what method
    # we use for the pivot matrix
    schur_rhs = rhs_reduced - B' * Cinv_rhs_pivot
    schur_solver = MadNLPHSL.Ma57Solver(S)
    MadNLP.factorize!(schur_solver)
    x = copy(schur_rhs)
    MadNLP.solve!(schur_solver, x)

    if use_hsl
        rhs_pivot_corr = rhs_pivot - B * x
        y = copy(rhs_pivot_corr)
        MadNLP.solve!(pivot_solver, y)
    else
        # B_perm has permuted rows, not columns. Multiplying by x is still valid
        rhs_pivot_corr = rhs_pivot_perm - B_perm * x
        y_perm = LinearAlgebra.LowerTriangular(C_perm) \ rhs_pivot_corr
        y = y_perm[invperm(colorder), :]
    end

    sol_schur = zeros(N, nrhs)
    sol_schur[R, :] .= x
    sol_schur[P, :] .= y

    K_full = kkt_matrix + kkt_matrix' - LinearAlgebra.Diagonal(kkt_matrix)
    residual = rhs - K_full * sol_schur
    res_norm = LinearAlgebra.norm(residual, Inf)
    println("CPU Schur residual (Inf): $res_norm")
    #@test res_norm <= 1e-6

    ma57 = MadNLPHSL.Ma57Solver(kkt_matrix)
    MadNLP.factorize!(ma57)
    inertia = MadNLP.inertia(ma57)
    println("KKT inertia: $inertia")
    sol_ma57 = copy(rhs)
    MadNLP.solve!(ma57, sol_ma57)
    maxdiff = maximum(abs.(sol_schur - sol_ma57))
    println("CPU Schur vs MA57 max error: $maxdiff")
    #@test maxdiff <= 1e-6
    return
end

@testset "basic-cuda" begin
    #test_cuda_linearsolve_synthetic()
    test_cpu_construct_schur_synthetic(; use_hsl = true)
    #test_cuda_construct_schur_synthetic()
    #test_cuda_construct_schur_synthetic(; sparse = true)
end
