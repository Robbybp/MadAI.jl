import MadNLP
import MadAI
import Random
import LinearAlgebra
import SparseArrays
import NLPModels

include("JuMP/models.jl")
include("solve-kkt.jl")

const MODELNAME = "mnist"
const NODES = 512
const LAYERS = 4
const ITERATE_SET = "first"

function test_kkt_residuals(nlp, LinearSolver, opt, iterate)
    # This comes from the new construction method
    kkt, kkt_rhs = iterate_to_kkt(nlp, iterate)
    #kkt_rhs = copy(kkt_rhs)
    N, M = size(kkt)
    linear_solver = LinearSolver(kkt; opt)

    Random.seed!(0)
    rhs_list = [kkt_rhs, zeros(N), ones(N), rand(N)]
    rhs_names = ["MadNLP", "zeros", "ones", "random"]

    MadNLP.factorize!(linear_solver)

    full_matrix, tril_to_full_view = MadNLP.get_tril_to_full(kkt)
    full_matrix.nzval .= tril_to_full_view

    residuals = []
    for rhs in rhs_list
        sol = copy(rhs)
        MadNLP.solve!(linear_solver, sol)
        refine_res = MadAI.refine!(
            sol,
            linear_solver,
            rhs;
            max_iter = 2,
            tol = 1e-5,
            full_matrix,
            tril_to_full_view,
        )
        resid = full_matrix * sol - rhs
        push!(residuals, resid)
    end

    for i in eachindex(rhs_names)
        str = rpad(rhs_names[i], maximum(length.(rhs_names)))
        maxerr = maximum(abs.(residuals[i]))
        println("$str: $(maxerr)")
    end
end

filename = "$MODELNAME-$(NODES)nodes$(LAYERS)layers-$ITERATE_SET.json"
fpath = joinpath(@__DIR__, "data", "iterates", filename)
iterate_data = open(fpath, "r") do io
    JSON.parse(io)
end

iterates = iterate_data["iterates"]
iterate = Dict(iterates[1])

model, formulation = get_model(MODELNAME, NODES, LAYERS)

nlp = NLPModelsJuMP.MathOptNLPModel(model)

pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
pivot_indices = MadAI.get_kkt_indices(model, pivot_vars, pivot_cons)
pivot_indices = convert(Vector{Int32}, pivot_indices)
blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
opt_linear_solver = MadAI.SchurComplementOptions(;
    ReducedSolver = MadNLPHSL.Ma57Solver,
    PivotSolver = MadAI.BlockTriangularSolver,
    pivot_indices,
    pivot_solver_opt,
)

kkt, kkt_rhs = iterate_to_kkt(nlp, iterate)
nvar = NLPModels.get_nvar(nlp)
kkt[1:nvar, 1:nvar] .+= 0.01
N, M = size(kkt)
pivot_dim = length(pivot_indices)
index_set = Set(pivot_indices)
reduced_indices = filter(i -> !(i in index_set), 1:N)
reduced_dim = length(reduced_indices)
orig_rhs_reduced = kkt_rhs[reduced_indices]
orig_rhs_pivot = kkt_rhs[pivot_indices]
P = pivot_indices
R = reduced_indices
A = kkt[R, R]
B = kkt[P, R] + kkt[R, P]'
C = kkt[P, P]

full_C, full_view_C = MadNLP.get_tril_to_full(C)
full_C.nzval .= full_view_C
pivot_solver = MadAI.BlockTriangularSolver(C; opt = pivot_solver_opt)
MadNLP.factorize!(pivot_solver)
# S = A - B' C \ B
# I want the error in C \ B
B_nz_cols = filter(i -> B.colptr[i] < B.colptr[i+1], 1:length(R))
compressed_B = Matrix(B[:, B_nz_cols])
compressed_sol = copy(compressed_B)
MadNLP.solve!(pivot_solver, compressed_sol)
resid = full_C * compressed_sol - compressed_B
maxerr = maximum(abs.(resid))
println("ϵ C \\ B (before refinement) = $maxerr")
#refine_res = MadAI.refine!(
#    compressed_sol,
#    pivot_solver,
#    compressed_B;
#    max_iter = 0,
#    tol = 1e-5,
#    full_matrix = full_C,
#    tril_to_full_view = full_view_C,
#)
#resid = full_C * compressed_sol - compressed_B
#maxerr = maximum(abs.(resid))
#println("$(refine_res.iterations) iterations of refinement")
#println("ϵ (after refinement)  = $maxerr")

BTCB = SparseArrays.spzeros(reduced_dim, reduced_dim)
BTCB[B_nz_cols, B_nz_cols] = LinearAlgebra.tril(compressed_B' * compressed_sol)
S = A - BTCB

full_S, full_view_S = MadNLP.get_tril_to_full(S)
full_S.nzval .= full_view_S

opt = MadNLP.default_options(MadNLPHSL.Ma57Solver)
# This actually makes things worse
opt.ma57_pivtol = 1e-10
schur_solver = MadNLPHSL.Ma57Solver(S; opt)
MadNLP.factorize!(schur_solver)

ra = kkt_rhs[reduced_indices]
rc = kkt_rhs[pivot_indices]

temp = copy(rc)
MadNLP.factorize!(pivot_solver)
MadNLP.solve!(pivot_solver, temp)

rs = ra - B' * temp
xa = copy(rs)
MadNLP.solve!(schur_solver, xa)
resid = full_S * xa - rs
maxerr = maximum(abs.(resid))
println("S \\ rs error (before refinement)  = $maxerr")
refine_res = MadAI.refine!(
    xa,
    schur_solver,
    rs;
    max_iter = 10,
    tol = 1e-5,
    full_matrix = full_S,
    tril_to_full_view = full_view_S,
)
resid = full_S * xa - rs
maxerr = maximum(abs.(resid))
println("$(refine_res.iterations) iterations of refinement")
println("ϵ (after refinement)  = $maxerr")
cond = LinearAlgebra.cond(Matrix(full_S))
println("cond(S): $cond")

#resid = full_C * temp - rc
#maxerr = maximum(abs.(resid))
#println("ϵ (before refinement)  = $maxerr")
#refine_res = MadAI.refine!(
#    temp,
#    pivot_solver,
#    rc;
#    max_iter = 10,
#    tol = 1e-5,
#    full_matrix = full_C,
#    tril_to_full_view = full_view_C,
#)
#resid = full_C * temp - rc
#maxerr = maximum(abs.(resid))
#println("$(refine_res.iterations) iterations of refinement")
#println("ϵ (after refinement)  = $maxerr")

# I don't need the RHS for now. First, I want to inspect the Schur complement
#temp = copy(orig_rhs_pivot)
#MadNLP.solve!(pivot_solver, temp)
#rhs_reduced = orig_rhs_reduced - B' * temp
