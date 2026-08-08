import ArgParse
import JuMP
import HSL_jll
import Ipopt
import MadNLP
import MadNLPHSL
import PythonCall

include("nn/nn.jl")
include("JuMP/models.jl")

const MADNLP_LINEAR_SOLVERS = Dict(
    "ma27" => MadNLPHSL.Ma27Solver,
    "ma57" => MadNLPHSL.Ma57Solver,
    "ma86" => MadNLPHSL.Ma86Solver,
)

function parse_commandline()
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
        "--solver"
            help = "NLP solver: madnlp or ipopt"
            default = "ipopt"
        "--linear-solver"
            help = "HSL linear solver: ma27, ma57, or ma86"
            default = "ma57"
    end
    args = ArgParse.parse_args(settings)
    args["solver"] = lowercase(args["solver"])
    args["linear-solver"] = lowercase(args["linear-solver"])
    if args["solver"] ∉ ("madnlp", "ipopt")
        error("Unknown solver: $(args["solver"])")
    end
    return args
end

function get_optimizer(solver, linear_solver)
    if solver == "madnlp"
        return JuMP.optimizer_with_attributes(
            MadNLP.Optimizer,
            "linear_solver" => MADNLP_LINEAR_SOLVERS[linear_solver],
            "tol" => 1e-6,
        )
    end
    return JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "linear_solver" => linear_solver,
        "tol" => 1e-6,
    )
end

# TODO: Expose these in CLI
modelname = "mnist"
nodes = 128
layers = 4

args = parse_commandline()
model, formulation = get_model(modelname, nodes, layers)

JuMP.set_optimizer(model, get_optimizer(args["solver"], args["linear-solver"]))
JuMP.optimize!(model)
