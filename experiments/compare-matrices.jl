import JSON
import NLPModelsJuMP
import MadNLP, MadNLPHSL
import MadAI

include("JuMP/models.jl")
include("solve-kkt.jl")
include("solve-kkt-old.jl")

const MODELNAME = "mnist"
const NODES = 2048
const LAYERS = 4
const ITERATE_SET = "first"

filename = "$MODELNAME-$(NODES)nodes$(LAYERS)layers-$ITERATE_SET-old.json"
fpath = joinpath(@__DIR__, "data", "iterates", filename)
iterate_data = open(fpath, "r") do io
    JSON.parse(io)
end

iterates = iterate_data["iterates"]
iterate = Dict(iterates[1])

model, formulation = get_model(MODELNAME, NODES, LAYERS)
nlp = NLPModelsJuMP.MathOptNLPModel(model)
LinearSolver = MadNLPHSL.Ma57Solver
opt_linear_solver = MadNLP.default_options(LinearSolver)
pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
pivot_indices = MadAI.get_kkt_indices(model, pivot_vars, pivot_cons)
pivot_indices = convert(Vector{Int32}, pivot_indices)
blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
madnlp_opt = MadAI.SchurComplementOptions(;
    ReducedSolver = MadNLPHSL.Ma57Solver,
    PivotSolver = MadAI.BlockTriangularSolver,
    pivot_indices,
    pivot_solver_opt,
)
fields = fieldnames(typeof(madnlp_opt))
madnlp_opt = Dict(zip(fields, getproperty.(Ref(madnlp_opt), fields)))

# This comes from the new construction method
kkt, rhs = iterate_to_kkt(nlp, iterate)

# The actual iterate we pass here doesn't matter. We just use its length.
results = solve_kkt_old(
    nlp, LinearSolver, opt_linear_solver, [iterate];
    madnlp_opt,
    return_iterates = true,
    return_kkt = true,
)
old_kkt = results[1].kkt
old_rhs = results[1].rhs

@assert all(kkt.rowval .== old_kkt.rowval)
@assert all(kkt.colptr .== old_kkt.colptr)
