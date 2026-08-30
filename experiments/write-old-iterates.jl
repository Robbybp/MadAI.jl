import JSON
import JuMP
import LinearAlgebra
import DataFrames
import MadAI
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import Statistics

include("solve-kkt.jl")
include("solve-kkt-old.jl")
include("JuMP/models.jl")

const MODELNAME = "mnist"
const NODES = 2048
const LAYERS = 4
const ITERATE_SET = "first"

function old_iterate_linear_solver_options(LinearSolver, model, formulation)
    LinearSolver === MadNLPHSL.Ma57Solver && return MadNLP.default_options(LinearSolver)

    pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
    pivot_indices = convert(
        Vector{Int32}, MadAI.get_kkt_indices(model, pivot_vars, pivot_cons),
    )
    blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
    return MadAI.SchurComplementOptions(;
        ReducedSolver = MadNLPHSL.Ma57Solver,
        PivotSolver = MadAI.BlockTriangularSolver,
        pivot_indices,
        pivot_solver_opt,
    )
end

function summarize_old_iterate_results(results)
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

    baseline_time = only(
        (row.t_factorize + row.t_solve) / row.n_iterates
        for row in DataFrames.eachrow(summary) if row.LinearSolver === MadNLPHSL.Ma57Solver
    )
    summary.speedup = [
        row.LinearSolver === MadAI.SchurComplementSolver ?
        baseline_time / ((row.t_factorize + row.t_solve) / row.n_iterates) : NaN
        for row in DataFrames.eachrow(summary)
    ]
    return summary
end

function solve_new_kkt_with_old_iterates()
    model, formulation = get_model(MODELNAME, NODES, LAYERS)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    fpath = joinpath(
        @__DIR__, "data", "iterates",
        "$MODELNAME-$(NODES)nodes$(LAYERS)layers-$ITERATE_SET-old.json",
    )
    iterate_data = open(fpath, "r") do io
        JSON.parse(io)
    end
    variables, constraints = MadAI.get_var_con_order(model)
    @assert iterate_data["variables"] == string.(JuMP.index.(variables))
    @assert iterate_data["constraints"] == string.(JuMP.index.(constraints))

    results = NamedTuple[]
    for LinearSolver in (MadNLPHSL.Ma57Solver, MadAI.SchurComplementSolver)
        opt_linear_solver = old_iterate_linear_solver_options(LinearSolver, model, formulation)
        for result in solve_kkt(nlp, LinearSolver, opt_linear_solver, iterate_data["iterates"])
            metadata = (; nodes = NODES, layers = LAYERS, modelname = MODELNAME, LinearSolver)
            push!(results, merge(metadata, result))
        end
    end
    summary = summarize_old_iterate_results(results)
    display(summary)
    return summary
end

function compare_new_old_iterates()
    iterate_dir = joinpath(@__DIR__, "data", "iterates")
    filename = "$MODELNAME-$(NODES)nodes$(LAYERS)layers-$ITERATE_SET"
    new_data = open(joinpath(iterate_dir, "$filename.json"), "r") do io
        JSON.parse(io)
    end
    old_data = open(joinpath(iterate_dir, "$filename-old.json"), "r") do io
        JSON.parse(io)
    end
    @assert new_data["variables"] == old_data["variables"]
    @assert new_data["constraints"] == old_data["constraints"]
    @assert length(new_data["iterates"]) == length(old_data["iterates"])

    coordinate_names = Dict(
        "primal" => new_data["variables"],
        "dual" => new_data["constraints"],
        "Ldual" => new_data["variables"],
        "Udual" => new_data["variables"],
    )
    comparisons = NamedTuple[]
    for (iterate, (new, old)) in enumerate(zip(new_data["iterates"], old_data["iterates"]))
        for vector_name in ("primal", "dual", "Ldual", "Udual")
            println("Comparing $vector_name vector at iterate $iterate")
            difference = Float64.(new[vector_name]) .- Float64.(old[vector_name])
            names = coordinate_names[vector_name]
            println("length(difference) = $(length(difference))")
            println("length(names) = $(length(names))")
            #@assert length(difference) == length(names)
            indices = partialsortperm(
                abs.(difference), 1:min(10, length(difference)); rev = true,
            )
            largest_differences = [
                (; index, name = names[index], difference = difference[index]) for index in indices
            ]
            push!(comparisons, (; iterate, vector_name, norm = LinearAlgebra.norm(difference),
                largest_differences))
        end
    end
    return comparisons
end

function main()
    model, formulation = get_model(MODELNAME, NODES, LAYERS)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)

    fpath = joinpath(
        @__DIR__, "data", "iterates",
        "$MODELNAME-$(NODES)nodes$(LAYERS)layers-$ITERATE_SET.json",
    )
    iterate_data = open(fpath, "r") do io
        JSON.parse(io)
    end
    variables, constraints = MadAI.get_var_con_order(model)
    @assert iterate_data["variables"] == string.(JuMP.index.(variables))
    @assert iterate_data["constraints"] == string.(JuMP.index.(constraints))

    pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
    pivot_indices = convert(
        Vector{Int32}, MadAI.get_kkt_indices(model, pivot_vars, pivot_cons),
    )
    blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
    schur_opt = MadAI.SchurComplementOptions(;
        ReducedSolver = MadNLPHSL.Ma57Solver,
        PivotSolver = MadAI.BlockTriangularSolver,
        pivot_indices,
        pivot_solver_opt,
    )
    fields = fieldnames(typeof(schur_opt))
    madnlp_opt = Dict(zip(fields, map(f -> getproperty(schur_opt, f), fields)))

    results = solve_kkt_old(
        nlp,
        MadAI.SchurComplementSolver,
        schur_opt,
        #MadNLPHSL.Ma57Solver,
        #MadNLP.default_options(MadNLPHSL.Ma57Solver),
        iterate_data["iterates"];
        madnlp_opt,
        return_iterates = true,
    )

    output = Dict{String,Any}(
        "variables" => string.(JuMP.index.(variables)),
        "constraints" => string.(JuMP.index.(constraints)),
        "iterates" => getproperty.(results, :iterates),
    )
    output_path = joinpath(
        @__DIR__, "data", "iterates",
        "$MODELNAME-$(NODES)nodes$(LAYERS)layers-$ITERATE_SET-old.json",
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON.print(io, output, 1)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    #main()
    #results = compare_new_old_iterates()
    solve_new_kkt_with_old_iterates()
end
