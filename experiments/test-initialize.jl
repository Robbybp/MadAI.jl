import ArgParse
import DataFrames
import JSON
import JuMP
import MadAI
import MadNLP, MadNLPHSL
import MathOptInterface as MOI

include("JuMP/models.jl")

const MODELNAME = "mnist"
const LAYERS = 4
const ARGS_WHEN_INCLUDED = Dict(
    "nodes" => 2048,
    "mu-init" => 1e-4,
    "bound-push" => 1e-4,
)

function parse_commandline()
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
        "--nodes"
            help = "Number of nodes per hidden layer"
            arg_type = Int
            default = 2048
        "--mu-init"
            help = "Initial barrier parameter"
            arg_type = Float64
            default = 1e-5
        "--bound-push"
            help = "Distance used to push the initial point from variable bounds"
            arg_type = Float64
            default = 1e-5
    end
    return ArgParse.parse_args(settings)
end

function main(args)
    nodes = args["nodes"]
    mu_init = args["mu-init"]
    bound_push = args["bound-push"]

    mgb, _ = get_model(MODELNAME, nodes, LAYERS; gray_box = true)
    gb_iterate_fname = "$MODELNAME-$(nodes)nodes$(LAYERS)layers-gb-last.json"
    gb_iterate_fpath = joinpath(@__DIR__, "data", "iterates", gb_iterate_fname)
    iterate_data = open(gb_iterate_fpath, "r") do io
        JSON.parse(io)
    end

    gb_variable_indices = MOI.get(JuMP.backend(mgb), MOI.ListOfVariableIndices())
    gb_variables = [JuMP.VariableRef(mgb, index) for index in gb_variable_indices]
    @assert iterate_data["variables"] == string.(JuMP.index.(gb_variables))
    last_primal = iterate_data["iterates"][end]["primal"]
    gb_primal_values = Dict(zip(gb_variables, last_primal))

    madnlp = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "linear_solver" => MadNLPHSL.Ma57Solver,
        "tol" => 1e-6,
        "print_user_options" => "yes",
        "bound_push" => bound_push,
        "max_iter" => 30,
        "barrier" => MadNLP.MonotoneUpdate(; mu_init),
    )
    display(madnlp.params)

    gb_varnames = JuMP.name.(gb_variables)
    xstart_by_name = Dict{String,Float64}(zip(gb_varnames, last_primal))
    mfull, _ = get_model(MODELNAME, nodes, LAYERS; xstart = xstart_by_name)

    mfull_vars = JuMP.variable_by_name.(mfull, gb_varnames)
    nmissing = count(mfull_vars .== nothing)
    println("$nmissing variables missing")
    for (var, val) in zip(mfull_vars, last_primal)
        JuMP.set_start_value(var, val)
    end
    JuMP.set_optimizer(mfull, madnlp)
    JuMP.optimize!(mfull)

    results = DataFrames.DataFrame([(; nodes,
        mu_init,
        bound_push,
        iterations = MOI.get(JuMP.backend(mfull), MOI.BarrierIterations()),
    )])
    display(results)
    return results
end

args = abspath(PROGRAM_FILE) == (@__FILE__) ? parse_commandline() : ARGS_WHEN_INCLUDED
main(args)
