import ArgParse
import CSV
import DataFrames
import JSON
import JuMP
import MadAI
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import Statistics

include("solve-kkt.jl")
include("solve-kkt-old.jl")
include("JuMP/models.jl")

function parse_commandline()
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
        "experiment"
            help = "Experiment to run"
            required = true
        "--old"
            help = "Use the old KKT-construction method"
            action = :store_true
            default = false
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

function runtime_experiment(; old = false)
    first_results = NamedTuple[]
    last_results = NamedTuple[]
    for modelname in ("mnist",),
        (nodes, layers) in ((512, 4), (1024, 4), (2048, 4))
        model, formulation = get_model(modelname, nodes, layers)
        nlp = NLPModelsJuMP.MathOptNLPModel(model)
        for iterate_set in ("first", "last")
            if iterate_set == "last" && (old || nodes == 2048)
                continue
            end
            iterates = load_iterates(model, modelname, nodes, layers, iterate_set)
            for LinearSolver in (MadNLPHSL.Ma57Solver, MadAI.SchurComplementSolver)
                opt_linear_solver = get_linear_solver_options(LinearSolver, model, formulation)
                if old
                    madnlp_opt = get_linear_solver_options(
                        MadAI.SchurComplementSolver, model, formulation,
                    )
                    fields = fieldnames(typeof(madnlp_opt))
                    madnlp_opt = Dict(zip(fields, getproperty.(Ref(madnlp_opt), fields)))
                    iterate_results = solve_kkt_old(
                        nlp, LinearSolver, opt_linear_solver, iterates; madnlp_opt,
                    )
                else
                    iterate_results = solve_kkt(nlp, LinearSolver, opt_linear_solver, iterates)
                end

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

function write_runtime_results(first_results, last_results; old = false)
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    function write_results(results, filename)
        results_df = DataFrames.DataFrame(results)
        results_df.LinearSolver = string.(results_df.LinearSolver)
        CSV.write(joinpath(results_dir, filename), results_df)
    end

    if old
        write_results(first_results, "runtime-first-old.csv")
    else
        write_results(first_results, "runtime-first.csv")
        write_results(last_results, "runtime-last.csv")
    end
    return nothing
end

function summarize_results(results)
    isempty(results) && return DataFrames.DataFrame()
    results_df = DataFrames.DataFrame(results)
    aggregate_columns = [:t_factorize, :t_solve, :residual, :refine_iter, :refine_success]
    group_columns = setdiff(propertynames(results_df), aggregate_columns)
    summary = DataFrames.combine(
        DataFrames.groupby(results_df, group_columns),
        DataFrames.nrow => :n_iterates,
        :refine_success => sum => :refine_success,
        :t_factorize => sum => :t_factorize,
        :t_solve => sum => :t_solve,
        :residual => Statistics.mean => :residual,
        :refine_iter => Statistics.mean => :refine_iter,
    )

    key(row) = (row.modelname, row.nodes, row.layers)
    is_baseline(row) = row.LinearSolver in (MadNLPHSL.Ma57Solver, MadNLPHSL.Ma86Solver)
    baseline_times = Dict(
        key(row) => (row.t_factorize + row.t_solve) / row.n_iterates
        for row in DataFrames.eachrow(summary) if is_baseline(row)
    )
    speedup = fill(NaN, DataFrames.nrow(summary))
    for (i, row) in enumerate(DataFrames.eachrow(summary))
        if row.LinearSolver === MadAI.SchurComplementSolver
            schur_time = (row.t_factorize + row.t_solve) / row.n_iterates
            speedup[i] = get(baseline_times, key(row), NaN) / schur_time
        end
    end
    summary.speedup = speedup
    return summary
end

function main()
    args = parse_commandline()
    if args["experiment"] == "runtime"
        first_results, last_results = runtime_experiment(; old = args["old"])
        write_runtime_results(first_results, last_results; old = args["old"])
        return first_results, last_results
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
