import ArgParse
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

function parse_commandline()
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
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
        "--last-iterates"
            help = "Write the last N iterates. Default is to write the first N."
            action = :store_true
            default = false
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

#function madnlp_iterate_callback(model)
#    variables, constraints = MadAI.get_var_con_order(model)
#    iterates = Dict{String,Any}(
#        "variables" => JuMP.name.(variables),
#        "constraints" => JuMP.name.(constraints),
#        "iterates" => Any[],
#    )
#
#    function MyCallback(solver, status)
#        status isa MadNLP.UserCallbackRegular || return true
#        push!(iterates["iterates"], Dict{String,Any}(
#            "primal" => copy(MadNLP.primal(MadNLP.get_x(solver))),
#            "dual" => copy(MadNLP.get_y(solver)),
#            "Ldual" => copy(MadNLP.primal(MadNLP.get_zl(solver))),
#            "Udual" => copy(MadNLP.primal(MadNLP.get_zu(solver))),
#            "barrier" => MadNLP.get_mu(solver),
#        ))
#        return true
#    end
#    return iterates, MyCallback
#end

# TODO: Expose these in CLI
modelname = "mnist"
nodes = 128
layers = 4

args = parse_commandline()
model, formulation = get_model(modelname, nodes, layers)

JuMP.set_optimizer(model, OPTIMIZER_LOOKUP[args["solver"]])
JuMP.set_optimizer_attribute(model, LINEAR_SOLVER_LOOKUP[args["solver"], args["linear-solver"]])
# TODO: Linear solver options

if args["solver"] == "madnlp"
    cb = Callback()
    JuMP.set_optimizer_attribute(model, "intermediate_callback", cb)
end

JuMP.optimize!(model)

# Collect iterates from callback
if args["write-iterates"] >= 1 && args["solver"] == "madnlp"
    n_iter = args["write-iterates"]
    if n_iter < length(cp.iterates)
        error("$n_iter iterates requested, but solve only had $(length(cp.iterates)) iterations")
    end

    if args["last-iterates"]
        iter_start = length(cp.iterates) - n_iter
        iterates_to_write = cp.iterates[iter_start:end]
        suffix = "-last"
    else
        iterates_to_write = cp.iterates[1:n_iter]
        suffix = "-first"
    end
    variables, constraints = MadAI.get_var_con_order(model)
    iterate_data = Dict{String,Any}(
        "variables" => JuMP.name.(variables),
        "constraints" => JuMP.name.(constraints),
        "iterates" => iterates_to_write,
    )
    fname = "$modelname-$(nodes)nodes$(layers)layers$(suffix).json"
    fpath = joinpath(@__DIR__, "data", "iterates", fname)
end
