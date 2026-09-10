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
const INIT_FROM_GB_OPTIONS = Dict(
    ("mnist", 128, 4) => (; mu_init = 1e-2, bound_push = 1e-2),
    ("mnist", 512, 4) => (; mu_init = 1e-2, bound_push = 1e-2),
    ("mnist", 1024, 4) => (; mu_init = 7e-4, bound_push = 1e2),
    ("mnist", 2048, 4) => (; mu_init = 1e-3, bound_push = 1e-3),
    ("scopf", 500, 5) => (; mu_init = 1e-4, bound_push = 1e-4, tol = 1e-6),
    ("scopf", 1000, 7) => (; mu_init = 1e-6, bound_push = 1e-6, tol = 1e-6),
    ("scopf", 1500, 10) => (; mu_init = 1e-6, bound_push = 1e-6, tol = 1e-6),
    ("lsv", 2048, 3) => (; mu_init = 1e-6, bound_push = 1e-6, tol = 1e-6),
)
const OPT_LOOKUP = Dict(
    # Metis or exact minimum degree croak on these matrices.
    #
    # MA57 (in HSL_jll) is compiled against LBT. So number of threads for L3 BLAS should
    # be controlled by lbt_set_num_threads above.
    ("madnlp", "ma27") => MadNLP.default_options(MadNLPHSL.Ma27Solver),
    ("madnlp", "ma57") => MadNLPHSL.Ma57Options(; ma57_pivot_order = 2), # In MA57, 2=AMD
    # I don't set ma86_num_threads because I'm not interested in benchmarking here. I'm just interested
    # in speed.
    ("madnlp", "ma86") => MadNLPHSL.Ma86Options(; ma86_order = MadNLPHSL.AMD), # ma86_num_threads = 1),
    ("madnlp", "ma97") => MadNLPHSL.Ma97Options(; ma97_order = MadNLPHSL.AMD), # ma97_num_threads = 1),
    ("ipopt", "ma27") => (;),
    ("ipopt", "ma57") => (; ma57_pivot_order = 2),
    ("ipopt", "ma86") => (; ma86_order = "amd"),
)
const LINEAR_SOLVER_BY_PROBLEM = Dict(
    "mnist" => "ma57",
    "scopf" => "ma86",
    "lsv" => "ma86",
)
ARGS_WHEN_INCLUDED = Dict(
    "modelname" => "mnist",
    "nodes" => 128,
    "layers" => 4,
    "solver" => "madnlp",
    "linear-solver" => nothing,
    "write-iterates" => 10,
    "gray-box" => false,
    "initialize-from-gb" => false,
    #"mu-init" => nothing,
    #"bound-push" => nothing,
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
            help = "HSL linear solver: ma27, ma57, or ma86 (default is model dependent)"
        "--write-iterates"
            help = "Number of iterates to write. Default, 0, writes nothing."
            arg_type = Int
            default = 0
        "--gray-box"
            help = "Construct the gray-box model"
            action = :store_true
        "--initialize-from-gb"
            help = "Initialize from the last saved iterate from the gray-box model"
            action = :store_true
        #"--mu-init"
        #    help = "Initial barrier parameter when initializing from the gray-box model"
        #    arg_type = Float64
        #"--bound-push"
        #    help = "Bound push when initializing from the gray-box model"
        #    arg_type = Float64
    end
    args = ArgParse.parse_args(settings)
    args["solver"] = lowercase(args["solver"])
    if args["linear-solver"] === nothing
        args["linear-solver"] = LINEAR_SOLVER_BY_PROBLEM[args["modelname"]]
    else
        args["linear-solver"] = lowercase(args["linear-solver"])
    end
    if args["solver"] ∉ ("madnlp", "ipopt")
        error("Unknown solver: $(args["solver"])")
    end
    return args
end

struct Callback <: MadNLP.AbstractUserCallback
    iterates::Vector{Any}
    Callback() = new([])
end

function (cb::Callback)(solver::MadNLP.AbstractMadNLPSolver, mode)
    push!(cb.iterates, Dict{String,Any}(
        # Does this contain primals + slacks???
        "primal" => copy(MadNLP.primal(MadNLP.get_x(solver))),
        "dual" => copy(MadNLP.get_y(solver)),
        "Ldual" => copy(MadNLP.primal(MadNLP.get_zl(solver))),
        "Udual" => copy(MadNLP.primal(MadNLP.get_zu(solver))),
        "barrier" => MadNLP.get_mu(solver),
    ))
    return true
end

function get_xstart_from_gb(modelname, nodes, layers)
    mgb, _ = get_model(modelname, nodes, layers; gray_box = true)
    fname = "$modelname-$(nodes)nodes$(layers)layers-gb-last.json"
    fpath = joinpath(@__DIR__, "data", "iterates", fname)
    iterate_data = open(fpath, "r") do io
        JSON.parse(io)
    end
    variables = JuMP.all_variables(mgb)
    @assert iterate_data["variables"] == string.(JuMP.index.(variables))
    last_primal = iterate_data["iterates"][end]["primal"]
    varnames = JuMP.name.(variables)
    values_by_name = Dict{String,Float64}(zip(varnames, last_primal))
    return values_by_name
end

args = abspath(PROGRAM_FILE) == (@__FILE__) ? parse_commandline() : ARGS_WHEN_INCLUDED
modelname = args["modelname"]
nodes = args["nodes"]
layers = args["layers"]
if args["linear-solver"] === nothing
    args["linear-solver"] = LINEAR_SOLVER_BY_PROBLEM[modelname]
end
println("ARGS:")
display(args)
xstart = args["initialize-from-gb"] ? get_xstart_from_gb(modelname, nodes, layers) : nothing
model, formulation = get_model(
    modelname, nodes, layers; gray_box = args["gray-box"], xstart,
)

JuMP.set_optimizer(model, OPTIMIZER_LOOKUP[args["solver"]])
JuMP.set_optimizer_attribute(model, "linear_solver", LINEAR_SOLVER_LOOKUP[args["solver"], args["linear-solver"]])
linear_solver_options = OPT_LOOKUP[args["solver"], args["linear-solver"]]
for field in fieldnames(typeof(linear_solver_options))
    JuMP.set_optimizer_attribute(model, string(field), getproperty(linear_solver_options, field))
end
#JuMP.set_optimizer_attributes(model, "max_iter" => 3000)
JuMP.set_optimizer_attributes(model, "max_iter" => 6000)
JuMP.set_optimizer_attributes(model, "tol" => 1e-6)
if args["initialize-from-gb"]
    init_options = INIT_FROM_GB_OPTIONS[(modelname, nodes, layers)]
    #mu_init = something(args["mu-init"], init_options.mu_init)
    #bound_push = something(args["bound-push"], init_options.bound_push)
    JuMP.set_optimizer_attribute(model, "bound_push", init_options.bound_push)
    if :tol in fieldnames(typeof(init_options))
        JuMP.set_optimizer_attribute(model, "tol", init_options.tol)
    end
    if args["solver"] == "madnlp"
        JuMP.set_optimizer_attribute(model, "barrier", MadNLP.MonotoneUpdate(; init_options.mu_init))
    elseif args["solver"] == "ipopt"
        JuMP.set_optimizer_attribute(model, "mu_init", init_options.mu_init)
    end
end
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
        gray_box_suffix = args["gray-box"] ? "-gb" : ""
        fname = "$modelname-$(nodes)nodes$(layers)layers$(gray_box_suffix)$(suffix).json"
        println("Writing $n_iter iterates to $fname")
        fpath = joinpath(@__DIR__, "data", "iterates", fname)
        mkpath(dirname(fpath))
        open(fpath, "w") do io
            JSON.print(io, iterate_data, 1)
        end
    end
end
