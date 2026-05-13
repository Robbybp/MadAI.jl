import MadNLP
import SparseArrays
import LinearAlgebra
import CUDA
import CUDA.CUSPARSE: CuSparseMatrixCSR
import CUDA.CUSOLVER

# Ordering that exposes the lower-triangular structure of the pivot matrix.
function _pivot_lowertri_order(n::Integer)
    @assert iseven(n)
    ny = Int(n / 2)
    rp = zeros(Int, n)
    cp = zeros(Int, n)
    for i in 1:ny
        cp[i] = i
        cp[i + ny] = 2 * ny - i + 1
        rp[i] = i + ny
        rp[i + ny] = ny - i + 1
    end
    return rp, cp
end

mutable struct GpuSchurComplementOptions{INT} <: MadNLP.AbstractOptions
    pivot_indices::Vector{INT}
    GpuSchurComplementOptions(; pivot_indices = Int32[]) = new{eltype(pivot_indices)}(pivot_indices)
end

mutable struct GpuSchurComplementSolver{T,INT} <: MadNLP.AbstractLinearSolver{T}
    csc::SparseArrays.SparseMatrixCSC{T,INT}
    pivot_indices::Vector{INT}
    reduced_indices::Vector{INT}
    roworder::Vector{Int}
    colorder::Vector{Int}
    C_gpu::Union{Nothing,CuSparseMatrixCSR{T}}
    LT_gpu::Union{Nothing,LinearAlgebra.LowerTriangular}
    B_gpu::Union{Nothing,CUDA.CuMatrix{T}}
    B_perm_gpu::Union{Nothing,CUDA.CuMatrix{T}}
    A_gpu::Union{Nothing,CUDA.CuMatrix{T}}
    temp_gpu::Union{Nothing,CUDA.CuMatrix{T}}
    S_gpu::Union{Nothing,CUDA.CuMatrix{T}}
    F_gpu::Union{Nothing,CUDA.CuMatrix{T}}
    ipiv_gpu::Union{Nothing,CUDA.CuVector{Int64}}
end

function GpuSchurComplementSolver(
    csc::SparseArrays.SparseMatrixCSC{T,INT};
    opt::GpuSchurComplementOptions = GpuSchurComplementOptions(),
    logger::MadNLP.MadNLPLogger = MadNLP.MadNLPLogger(),
) where {T,INT}
    pivot_indices = opt.pivot_indices
    isempty(pivot_indices) && error("GpuSchurComplementSolver requires pivot_indices.")
    pivot_dim = length(pivot_indices)
    iseven(pivot_dim) || error("pivot_indices length must be even.")
    index_set = Set(pivot_indices)
    reduced_indices = convert(Vector{INT}, filter(i -> !(i in index_set), 1:csc.n))
    roworder, colorder = _pivot_lowertri_order(pivot_dim)
    return GpuSchurComplementSolver{T,INT}(
        csc,
        pivot_indices,
        reduced_indices,
        roworder,
        colorder,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

MadNLP.input_type(::Type{GpuSchurComplementSolver}) = :csc
MadNLP.default_options(::Type{GpuSchurComplementSolver}) = GpuSchurComplementOptions()
MadNLP.is_supported(::Type{GpuSchurComplementSolver}, ::Type{T}) where {T <: AbstractFloat} = true
MadNLP.is_inertia(solver::GpuSchurComplementSolver) = solver.ipiv_gpu !== nothing

function MadNLP.introduce(solver::GpuSchurComplementSolver)
    pivot_dim = length(solver.pivot_indices)
    return "GPU Schur-complement solver (pivot size $(pivot_dim)x$(pivot_dim))"
end

function MadNLP.factorize!(solver::GpuSchurComplementSolver{T,INT}) where {T,INT}
    P = solver.pivot_indices
    R = solver.reduced_indices
    csc = solver.csc

    A = Matrix(csc[R, R])
    B = Matrix(csc[P, R] + csc[R, P]')
    C = csc[P, P]
    C_full = C + C' - LinearAlgebra.Diagonal(C)

    solver.roworder, solver.colorder = _pivot_lowertri_order(size(C_full, 1))
    roworder = solver.roworder
    colorder = solver.colorder

    C_perm = C_full[roworder, colorder]
    B_perm = B[roworder, :]

    solver.C_gpu = CuSparseMatrixCSR(C_perm)
    solver.LT_gpu = LinearAlgebra.LowerTriangular(solver.C_gpu)
    solver.B_gpu = CUDA.CuMatrix(B)
    solver.B_perm_gpu = CUDA.CuMatrix(B_perm)
    solver.A_gpu = CUDA.CuMatrix(A)
    solver.temp_gpu = CUDA.CuMatrix(copy(B_perm))

    LinearAlgebra.ldiv!(solver.temp_gpu, solver.LT_gpu, solver.B_perm_gpu)
    temp_unperm = solver.temp_gpu[invperm(colorder), :]
    BTCB_gpu = solver.B_gpu' * temp_unperm

    solver.S_gpu = CUDA.CuMatrix(solver.A_gpu)
    solver.S_gpu .= solver.A_gpu .- BTCB_gpu
    solver.S_gpu .= LinearAlgebra.tril(solver.S_gpu)

    F_gpu, ipiv_gpu, _ = CUSOLVER.sytrf!('L', solver.S_gpu)
    solver.F_gpu = F_gpu
    solver.ipiv_gpu = CUDA.CuVector{Int64}(ipiv_gpu)
    return solver
end

function MadNLP.solve!(solver::GpuSchurComplementSolver{T,INT}, rhs::Vector{T}) where {T,INT}
    solver.F_gpu === nothing && error("Call factorize! before solve!.")
    P = solver.pivot_indices
    R = solver.reduced_indices
    roworder = solver.roworder
    colorder = solver.colorder

    rhs_reduced = rhs[R]
    rhs_pivot_perm = rhs[P][roworder]

    rhs_reduced_gpu = CUDA.CuVector(rhs_reduced)
    rhs_pivot_perm_gpu = CUDA.CuVector(rhs_pivot_perm)

    Cinv_rhs_pivot_gpu = CUDA.CuVector(copy(rhs_pivot_perm_gpu))
    LinearAlgebra.ldiv!(Cinv_rhs_pivot_gpu, solver.LT_gpu, rhs_pivot_perm_gpu)
    Cinv_rhs_pivot_gpu = Cinv_rhs_pivot_gpu[invperm(colorder)]

    schur_rhs_gpu = rhs_reduced_gpu - solver.B_gpu' * Cinv_rhs_pivot_gpu
    x_gpu = CUDA.CuVector(copy(schur_rhs_gpu))
    CUSOLVER.sytrs!('L', solver.F_gpu, solver.ipiv_gpu, x_gpu)

    rhs_pivot_corr_gpu = rhs_pivot_perm_gpu - solver.B_perm_gpu * x_gpu
    y_perm_gpu = CUDA.CuVector(copy(rhs_pivot_corr_gpu))
    LinearAlgebra.ldiv!(y_perm_gpu, solver.LT_gpu, rhs_pivot_corr_gpu)

    rhs[R] .= Vector(x_gpu)
    rhs[P] .= Vector(y_perm_gpu[invperm(colorder)])
    return rhs
end

MadNLP.improve!(::GpuSchurComplementSolver) = false

function MadNLP.inertia(solver::GpuSchurComplementSolver)
    solver.ipiv_gpu === nothing && error("Call factorize! before inertia.")
    F = Array(solver.F_gpu)
    ipiv = Vector(solver.ipiv_gpu)
    n = length(ipiv)
    pos = zero(Int)
    zero_count = zero(Int)
    neg = zero(Int)
    tol = eps(real(eltype(F))) * max(1.0, LinearAlgebra.opnorm(F, 1))
    k = 1
    while k <= n
        pk = ipiv[k]
        if pk > 0
            d = F[k, k]
            if abs(d) <= tol
                zero_count += 1
            elseif d > 0
                pos += 1
            else
                neg += 1
            end
            k += 1
        else
            d11 = F[k, k]
            d22 = F[k + 1, k + 1]
            d12 = F[k + 1, k]
            tr = d11 + d22
            det = d11 * d22 - d12 * d12
            disc = max(tr * tr - 4 * det, zero(eltype(tr)))
            λ1 = (tr + sqrt(disc)) / 2
            λ2 = (tr - sqrt(disc)) / 2
            for λ in (λ1, λ2)
                if abs(λ) <= tol
                    zero_count += 1
                elseif λ > 0
                    pos += 1
                else
                    neg += 1
                end
            end
            k += 2
        end
    end
    return (pos, zero_count, neg)
end
