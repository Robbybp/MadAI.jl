import JuMP
import MadAI
import MadNLP
import MadNLPHSL
import NLPModelsJuMP

include("JuMP/models.jl")

modelname = "scopf"
nodes = 500
layers = 5

model, formulation = get_model(modelname, nodes, layers)

pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
for var in pivot_vars
    if JuMP.has_lower_bound(var) && JuMP.has_upper_bound(var) && JuMP.lower_bound(var) == JuMP.upper_bound(var)
        JuMP.delete(model, JuMP.LowerBoundRef(var))
        JuMP.delete(model, JuMP.UpperBoundRef(var))
    end
end

nlp = NLPModelsJuMP.MathOptNLPModel(model)
madnlp = MadNLP.MadNLPSolver(nlp; fixed_variable_treatment=MadNLP.RelaxBound)
MadNLP.initialize!(madnlp)
matrix = MadNLP.get_kkt(MadNLP.get_kkt(madnlp))

pivot_indices = convert(Vector{Int32}, MadAI.get_kkt_indices(model, pivot_vars, pivot_cons))
blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
opt = MadAI.SchurComplementOptions(;
    ReducedSolver = MadNLPHSL.Ma57Solver,
    PivotSolver = MadAI.BlockTriangularSolver,
    pivot_indices,
    pivot_solver_opt,
)

linear_solver = MadAI.SchurComplementSolver(matrix; opt)
