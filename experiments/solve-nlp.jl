import ArgParse
import JSON
import JuMP
import HSL_jll
import Ipopt
import MadAI
import MadNLP
import MadNLPHSL
import PythonCall

include("nn/nn.jl")
include("JuMP/models.jl")

# TODO: Custom Schur complement solver?
const LINEAR_SOLVER_LOOKUP = Dict(
    ("madnlp", "ma27") => MadNLPHSL.Ma27Solver,
    ("madnlp", "ma57") => MadNLPHSL.Ma57Solver,
    ("madnlp", "ma86") => MadNLPHSL.Ma86Solver,
    ("ipopt", "ma27") => "ma27",
    ("ipopt", "ma57") => "ma57",
    ("ipopt", "ma86") => "ma86",
)
const OPTIMIZER_LOOKUP = Dict(
    "ipopt" => Ipopt.Optimizer,
    "madnlp" => MadNLP.Optimizer,
)
ARGS_WHEN_INCLUDED = Dict(
    "modelname" => "mnist",
    "nodes" => 128,
    "layers" => 4,
    "solver" => "madnlp",
    "linear-solver" => "ma57",
    "write-iterates" => 10,
)

function parse_commandline()
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
        "modelname"
            help = "Model name"
            required = true
        "--nodes"
            help = "Number of nodes per hidden layer"
            arg_type = Int
            required = true
        "--layers"
            help = "Number of hidden layers"
            arg_type = Int
            required = true
        "--solver"
            help = "NLP solver: madnlp or ipopt"
            default = "ipopt"
        "--linear-solver"
            help = "HSL linear solver: ma27, ma57, or ma86"
            default = "ma57"
        "--write-iterates"
            help = "Number of iterates to write. Default, 0, writes nothing."
            arg_type = Int
            default = 0
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
            "acceptable_tol" => 1e-4,
        )
    end
    return JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "linear_solver" => linear_solver,
        "tol" => 1e-6,
        "acceptable_tol" => 1e-4,
    )
end

struct Callback <: MadNLP.AbstractUserCallback
    iterates::Vector{Any}
    Callback() = new([])
end

function (cb::Callback)(solver::MadNLP.AbstractMadNLPSolver, mode)
    push!(cb.iterates, Dict{String,Any}(
        "primal" => copy(MadNLP.primal(MadNLP.get_x(solver))),
        "dual" => copy(MadNLP.get_y(solver)),
        "Ldual" => copy(MadNLP.primal(MadNLP.get_zl(solver))),
        "Udual" => copy(MadNLP.primal(MadNLP.get_zu(solver))),
        "barrier" => MadNLP.get_mu(solver),
    ))
    return true
end

args = abspath(PROGRAM_FILE) == (@__FILE__) ? parse_commandline() : ARGS_WHEN_INCLUDED
modelname = args["modelname"]
nodes = args["nodes"]
layers = args["layers"]
model, formulation = get_model(modelname, nodes, layers)

JuMP.set_optimizer(model, OPTIMIZER_LOOKUP[args["solver"]])
JuMP.set_optimizer_attribute(model, "linear_solver", LINEAR_SOLVER_LOOKUP[args["solver"], args["linear-solver"]])
# TODO: Linear solver options

if args["solver"] == "madnlp"
    cb = Callback()
    JuMP.set_optimizer_attribute(model, "intermediate_callback", cb)
end

JuMP.optimize!(model)

# Collect iterates from callback
if args["write-iterates"] >= 1 && args["solver"] == "madnlp"
    n_iter = args["write-iterates"]
    if n_iter > length(cb.iterates)
        error("$n_iter iterates requested, but solve only had $(length(cb.iterates)) iterations")
    end

    variables, constraints = MadAI.get_var_con_order(model)
    for (suffix, iterates_to_write) in (
        ("-first", cb.iterates[1:n_iter]),
        ("-last", cb.iterates[end-n_iter+1:end]),
    )
        iterate_data = Dict{String,Any}(
            "variables" => string.(JuMP.index.(variables)),
            "constraints" => string.(JuMP.index.(constraints)),
            "iterates" => iterates_to_write,
        )
        fname = "$modelname-$(nodes)nodes$(layers)layers$(suffix).json"
        println("Writing $n_iter iterates to $fname")
        fpath = joinpath(@__DIR__, "data", "iterates", fname)
        mkpath(dirname(fpath))
        open(fpath, "w") do io
            JSON.print(io, iterate_data, 1)
        end
    end
end
