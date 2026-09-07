import ArgParse
import CSV
import DataFrames
import JSON
import JuMP
import MadAI
import MadNLP
import MadNLPHSL
import NLPModels
import NLPModelsJuMP
import PythonCall
import Statistics
import LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)
LinearAlgebra.BLAS.lbt_set_num_threads(1)

include("solve-kkt.jl")
include("solve-kkt-old.jl")
include("JuMP/models.jl")
include("write-latex.jl")

function parse_commandline()
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
        "experiment"
            help = "Experiment to run"
            required = true
        "--old"
            help = "Use the old KKT-construction method"
            action = :store_true
        "--dry-run"
            help = "Run the experiment without writing result files"
            action = :store_true
    end
    return ArgParse.parse_args(settings)
end

const HSL_SOLVER_BY_MODEL = Dict(
    "mnist" => MadNLPHSL.Ma57Solver,
    "scopf" => MadNLPHSL.Ma86Solver,
    "lsv" => MadNLPHSL.Ma86Solver,
)

const HSL_OPTIONS_LOOKUP = Dict(
    MadNLPHSL.Ma57Solver => MadNLPHSL.Ma57Options(; ma57_pivot_order = 2), # AMD
    MadNLPHSL.Ma86Solver => MadNLPHSL.Ma86Options(; ma86_order = MadNLPHSL.AMD, ma86_num_threads = 1),
)

function get_linear_solver_options(LinearSolver, model, formulation)
    if LinearSolver in values(HSL_SOLVER_BY_MODEL)
        return HSL_OPTIONS_LOOKUP[LinearSolver]
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
    println("Reading iterates from $fname")
    fpath = joinpath(@__DIR__, "data", "iterates", fname)
    iterate_data = open(fpath, "r") do io
        JSON.parse(io)
    end

    variables, constraints = MadAI.get_var_con_order(model)
    @assert iterate_data["variables"] == string.(JuMP.index.(variables))
    @assert iterate_data["constraints"] == string.(JuMP.index.(constraints))
    return iterate_data["iterates"]
end

const NN_BY_MODEL = Dict(
    "mnist" => [
        (512, 4),
        (1024, 4),
        (2048, 4),
    ],
    "scopf" => [
        (500, 5),
        (1000, 7),
        (1500, 10),
    ],
    "lsv" => [
        #(32, 3),
        (128, 3),
        (512, 3),
        (2048, 3),
    ]
)

const ACTIVATION_LABELS = Dict(
    "GELU" => "GELU",
    "ReLU" => "ReLU",
    "Sigmoid" => "Sigmoid",
    "Softmax" => "SoftMax",
    "Softplus" => "SoftPlus",
    "Tanh" => "Tanh",
)

function get_nn_structure(modelname, nodes, layers)
    nn = get_nn(modelname, nodes, layers)
    input_dim = nothing
    output_dim = nothing
    previous_linear_output = nothing
    hidden_layers = 0
    hidden_layer_widths = Int[]
    activations = String[]

    for layer in nn.children()
        layer_type = PythonCall.pyconvert(String, layer.__class__.__name__)
        if layer_type == "Linear"
            input_dim === nothing && (input_dim = PythonCall.pyconvert(Int, layer.in_features))
            output_dim = PythonCall.pyconvert(Int, layer.out_features)
            previous_linear_output = output_dim
        elseif haskey(ACTIVATION_LABELS, layer_type)
            layer_type == "Softmax" || begin
                hidden_layers += 1
                push!(hidden_layer_widths, previous_linear_output)
            end
            layer_label = ACTIVATION_LABELS[layer_type]
            layer_label ∈ activations || push!(activations, layer_label)
        end
    end

    trained_parameters = 0
    for parameter in nn.parameters()
        if PythonCall.pyconvert(Bool, parameter.requires_grad)
            trained_parameters += PythonCall.pyconvert(Int, parameter.numel())
        end
    end
    @assert !isempty(hidden_layer_widths)
    @assert all(==(first(hidden_layer_widths)), hidden_layer_widths)
    return (; model = uppercase(modelname), inputs = input_dim, outputs = output_dim,
        layer_width = first(hidden_layer_widths), layers = hidden_layers, trained_parameters,
        activations = join(activations, "+"))
end

function nn_structure_experiment()
    structures = NamedTuple[]
    for modelname in ("mnist", "scopf", "lsv"), (nodes, layers) in NN_BY_MODEL[modelname]
        push!(structures, get_nn_structure(modelname, nodes, layers))
    end
    return DataFrames.DataFrame(structures)
end

function problem_structure_experiment()
    structures = NamedTuple[]
    for modelname in ("mnist", "scopf", "lsv"), (nodes, layers) in NN_BY_MODEL[modelname]
        nn_structure = get_nn_structure(modelname, nodes, layers)
        model, _ = get_model(modelname, nodes, layers)
        nlp = NLPModelsJuMP.MathOptNLPModel(model)
        push!(structures, (;
            model = nn_structure.model,
            nodes,
            layers,
            trained_parameters = nn_structure.trained_parameters,
            nvar = NLPModels.get_nvar(nlp),
            ncon = NLPModels.get_ncon(nlp),
            nnzj = NLPModels.get_nnzj(nlp),
            nnzh = NLPModels.get_nnzh(nlp),
        ))
    end
    return DataFrames.DataFrame(structures)
end

function precompile_runtime_experiment()
    modelname = "mnist"
    nodes, layers = 128, 4
    model, formulation = get_model(modelname, nodes, layers)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    iterates = load_iterates(model, modelname, nodes, layers, "first")[1:1]

    println("Precompiling KKT solvers on MNIST $(nodes)-node, $(layers)-layer model")
    for LinearSolver in (HSL_SOLVER_BY_MODEL[modelname], MadAI.SchurComplementSolver)
        opt_linear_solver = get_linear_solver_options(LinearSolver, model, formulation)
        solve_kkt(nlp, LinearSolver, opt_linear_solver, iterates)
    end
    return nothing
end

function runtime_experiment(; old = false)
    first_results = NamedTuple[]
    last_results = NamedTuple[]
    for modelname in ["mnist", "scopf", "lsv"],
        (nodes, layers) in NN_BY_MODEL[modelname]
        model, formulation = get_model(modelname, nodes, layers)
        nlp = NLPModelsJuMP.MathOptNLPModel(model)
        for iterate_set in ("first", "last")
            #if iterate_set == "last" && (old || nodes == 2048)
            #    continue
            #end
            iterates = load_iterates(model, modelname, nodes, layers, iterate_set)
            for LinearSolver in (HSL_SOLVER_BY_MODEL[modelname], MadAI.SchurComplementSolver)
                println("MODEL = $modelname")
                println("$iterate_set $(length(iterates)) iterations")
                println("LinearSolver = $LinearSolver")
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
                    # MA57 hits an Int32 overflow with the largest MA86 model
                    MadNLPLinearSolver = modelname == "lsv" ? MadNLPHSL.Ma86Solver : MadNLPHSL.Ma57Solver
                    iterate_results = solve_kkt(nlp, LinearSolver, opt_linear_solver, iterates; MadNLPLinearSolver, ma86_order = MadNLPHSL.AMD)
                end

                metadata = (; nodes, layers, modelname, LinearSolver)
                target = iterate_set == "first" ? first_results : last_results
                for result in iterate_results
                    push!(target, merge(metadata, result))
                end
                display(iterate_results)
            end
        end
    end
    return first_results, last_results
end

function flops_experiment()
    results = NamedTuple[]
    for modelname in ("mnist", "scopf", "lsv")
        nodes, layers = last(NN_BY_MODEL[modelname])
        model, formulation = get_model(modelname, nodes, layers)
        nlp = NLPModelsJuMP.MathOptNLPModel(model)
        iterates = vcat(
            load_iterates(model, modelname, nodes, layers, "first"),
            load_iterates(model, modelname, nodes, layers, "last"),
        )
        HSLLinearSolver = HSL_SOLVER_BY_MODEL[modelname]
        hsl_options = get_linear_solver_options(HSLLinearSolver, model, formulation)
        schur_options = get_linear_solver_options(MadAI.SchurComplementSolver, model, formulation)
        matrix_results = benchmark_kkt_matrices(
            nlp, HSLLinearSolver, hsl_options, schur_options, iterates;
            ma86_order = MadNLPHSL.AMD,
        )
        metadata = (; modelname, nodes, layers, HSLLinearSolver)
        append!(results, merge.(Ref(metadata), matrix_results))
    end
    return summarize_flops_results(results)
end

function summarize_flops_results(results)
    results_df = DataFrames.DataFrame(results)
    group_columns = [:modelname, :nodes, :layers, :HSLLinearSolver, :matrix_type]
    summary = DataFrames.combine(
        DataFrames.groupby(results_df, group_columns),
        DataFrames.nrow => :n_iterates,
        :dim => first => :dim,
        :nnz => first => :nnz,
        :factor_size => Statistics.mean => :factor_nnz,
        :flops => Statistics.mean => :flops,
        :n2by2 => Statistics.mean => :n2by2,
        :t_factorize => Statistics.mean => :t_factorize,
        :t_solve => Statistics.mean => :t_solve,
        :residual => Statistics.mean => :residual,
        :refine_iter => Statistics.mean => :refine_iter,
        :refine_success => all => :refine_success,
    )
    summary.dim = Int.(summary.dim)
    return summary
end

function profile_schur_experiment()
    results = NamedTuple[]
    for modelname in ("mnist", "scopf", "lsv"),
        (nodes, layers) in NN_BY_MODEL[modelname]
        model, formulation = get_model(modelname, nodes, layers)
        nlp = NLPModelsJuMP.MathOptNLPModel(model)
        iterates = vcat(
            load_iterates(model, modelname, nodes, layers, "first"),
            load_iterates(model, modelname, nodes, layers, "last"),
        )
        opt_linear_solver = get_linear_solver_options(
            MadAI.SchurComplementSolver, model, formulation,
        )
        MadNLPLinearSolver = modelname == "lsv" ? MadNLPHSL.Ma86Solver : MadNLPHSL.Ma57Solver
        result = profile_schur(
            nlp, opt_linear_solver, iterates;
            MadNLPLinearSolver,
            ma86_order = MadNLPHSL.AMD,
        )
        push!(results, merge((; modelname, nodes, layers), result))
    end
    return DataFrames.DataFrame(results)
end

function write_runtime_results(first_results, last_results; old = false)
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    function write_results(results, filename)
        results_df = DataFrames.DataFrame(results)
        results_df.LinearSolver = string.(results_df.LinearSolver)
        fpath = joinpath(results_dir, filename)
        println("Saving results to $fpath")
        CSV.write(fpath, results_df)
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
    aggregate_columns = [:t_init, :t_factorize, :t_solve, :residual, :refine_iter, :refine_success]
    group_columns = setdiff(propertynames(results_df), aggregate_columns)
    summary = DataFrames.combine(
        DataFrames.groupby(results_df, group_columns),
        DataFrames.nrow => :n_iterates,
        :refine_success => sum => :refine_success,
        :t_init => Statistics.mean => :t_init,
        :t_factorize => sum => :t_factorize,
        :t_solve => sum => :t_solve,
        :residual => Statistics.mean => :residual,
        :refine_iter => Statistics.mean => :refine_iter,
    )

    key(row) = (row.modelname, row.nodes, row.layers)
    is_baseline(row) = string(row.LinearSolver) in (
        string(MadNLPHSL.Ma57Solver), string(MadNLPHSL.Ma86Solver),
    )
    baseline_times = Dict(
        key(row) => (row.t_factorize + row.t_solve) / row.n_iterates
        for row in DataFrames.eachrow(summary) if is_baseline(row)
    )
    speedup = fill(NaN, DataFrames.nrow(summary))
    for (i, row) in enumerate(DataFrames.eachrow(summary))
        if string(row.LinearSolver) == string(MadAI.SchurComplementSolver)
            schur_time = (row.t_factorize + row.t_solve) / row.n_iterates
            speedup[i] = get(baseline_times, key(row), NaN) / schur_time
        end
    end
    summary.speedup = speedup
    return summary
end

function main()
    args = parse_commandline()
    if args["experiment"] == "nn-structure"
        structures = nn_structure_experiment()
        results_dir = joinpath(@__DIR__, "results")
        mkpath(results_dir)
        csv_path = joinpath(results_dir, "nn-structure.csv")
        latex_path = joinpath(results_dir, "nn-structure.txt")
        CSV.write(csv_path, structures)
        write_nn_structure_latex_file(latex_path, structures)
        println(nn_structure_latex_contents(structures))
        println("Wrote $csv_path")
        println("Wrote $latex_path")
        return structures
    elseif args["experiment"] == "problem-structure"
        structures = problem_structure_experiment()
        results_dir = joinpath(@__DIR__, "results")
        mkpath(results_dir)
        csv_path = joinpath(results_dir, "problem-structure.csv")
        latex_path = joinpath(results_dir, "problem-structure.txt")
        CSV.write(csv_path, structures)
        write_problem_structure_latex_file(latex_path, structures)
        println(problem_structure_latex_contents(structures))
        println("Wrote $csv_path")
        println("Wrote $latex_path")
        return structures
    elseif args["experiment"] == "runtime"
        precompile_runtime_experiment()
        first_results, last_results = runtime_experiment(; old = args["old"])
        if args["dry-run"]
            println("Dry run: not writing result files")
        else
            write_runtime_results(first_results, last_results; old = args["old"])
        end
        return first_results, last_results
    elseif args["experiment"] == "flops"
        results = flops_experiment()
        results_dir = joinpath(@__DIR__, "results")
        mkpath(results_dir)
        fpath = joinpath(results_dir, "hsl-flops.csv")
        CSV.write(fpath, results)
        println("Saving results to $fpath")
        return results
    elseif args["experiment"] == "profile-schur"
        results = profile_schur_experiment()
        results_dir = joinpath(@__DIR__, "results")
        mkpath(results_dir)
        fpath = joinpath(results_dir, "profile-schur.csv")
        CSV.write(fpath, results)
        println("Saving results to $fpath")
        return results
    end
    error("Unknown experiment: $(args["experiment"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = main()
    if result isa DataFrames.DataFrame
        display(result)
    else
        first_results, last_results = result
        println("First iterates")
        println("--------------")
        display(summarize_results(first_results))
        println("Last iterates")
        println("-------------")
        display(summarize_results(last_results))
        println("Combined iterates")
        println("-----------------")
        display(summarize_results(vcat(first_results, last_results)))
    end
end
