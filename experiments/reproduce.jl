import ArgParse
import DataFrames
import JSON
import JuMP
import MadAI
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import Statistics

#include("solve-kkt.jl")
include("solve-kkt-old.jl")
include("JuMP/models.jl")

function parse_commandline()
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
        "experiment"
            help = "Experiment to run"
            required = true
    end
    return ArgParse.parse_args(settings)
end

function get_linear_solver_options(LinearSolver, model, formulation)
    if LinearSolver === MadNLPHSL.Ma57Solver
        return MadNLP.default_options(LinearSolver)
    end

    pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
    pivot_indices = MadAI.get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
    return MadAI.SchurComplementOptions(;
        ReducedSolver = MadNLPHSL.Ma57Solver,
        PivotSolver = MadAI.BlockTriangularSolver,
        pivot_indices,
        pivot_solver_opt,
    )
end

function load_iterates(model, modelname, nodes, layers, iterate_set)
    fname = "$modelname-$(nodes)nodes$(layers)layers-$iterate_set.json"
    fpath = joinpath(@__DIR__, "data", "iterates", fname)
    iterate_data = open(fpath, "r") do io
        JSON.parse(io)
    end

    variables, constraints = MadAI.get_var_con_order(model)
    @assert iterate_data["variables"] == string.(JuMP.index.(variables))
    @assert iterate_data["constraints"] == string.(JuMP.index.(constraints))
    return iterate_data["iterates"]
end

function runtime_experiment()
    first_results = NamedTuple[]
    last_results = NamedTuple[]
    for modelname in ("mnist",),
        (nodes, layers) in ((512, 4), (1024, 4))
        model, formulation = get_model(modelname, nodes, layers)
        nlp = NLPModelsJuMP.MathOptNLPModel(model)
        for iterate_set in ("first", "last")
            iterates = load_iterates(model, modelname, nodes, layers, iterate_set)
            for LinearSolver in (MadNLPHSL.Ma57Solver, MadAI.SchurComplementSolver)
                opt_linear_solver = get_linear_solver_options(LinearSolver, model, formulation)

                # When iterating with SchurComplementSolver, we always need to pass its options
                # to MadNLP
                madnlp_opt = get_linear_solver_options(MadAI.SchurComplementSolver, model, formulation)
                fields = fieldnames(typeof(madnlp_opt))
                madnlp_opt = Dict(zip(fields, map(f -> getproperty(madnlp_opt, f), fields)))
                # Note the updated call signature
                iterate_results = solve_kkt(nlp, LinearSolver, opt_linear_solver, iterates; madnlp_opt)

                metadata = (; nodes, layers, modelname, LinearSolver)
                target = iterate_set == "first" ? first_results : last_results
                for result in iterate_results
                    push!(target, merge(metadata, result))
                end
            end
        end
    end
    return first_results, last_results
end

function summarize_results(results)
    isempty(results) && return DataFrames.DataFrame()
    results_df = DataFrames.DataFrame(results)
    aggregate_columns = [:t_factorize, :t_solve, :residual, :refine_iter]
    group_columns = setdiff(propertynames(results_df), aggregate_columns)
    return DataFrames.combine(
        DataFrames.groupby(results_df, group_columns),
        :t_factorize => sum => :t_factorize,
        :t_solve => sum => :t_solve,
        :residual => Statistics.mean => :residual,
        :refine_iter => Statistics.mean => :refine_iter,
    )
end

function main()
    args = parse_commandline()
    if args["experiment"] == "runtime"
        return runtime_experiment()
    end
    error("Unknown experiment: $(args["experiment"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    first_results, last_results = main()
    println("First iterates")
    println("--------------")
    display(summarize_results(first_results))
    println("Last iterates")
    println("-------------")
    display(summarize_results(last_results))
end
