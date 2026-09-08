import ArgParse
import CSV
import DataFrames
import Statistics
using Printf: @sprintf


function nn_parameter_count(modelname, nodes, layers)
    if modelname == "mnist"
        return 785 * nodes + layers * (nodes^2 + nodes) + 10 * nodes + 10
    elseif modelname == "scopf"
        return 118 * nodes + (layers - 3) * (nodes^2 + nodes) + 38 * nodes + 37
    elseif modelname == "lsv"
        # The fixed normalization layer is not counted as an NN parameter.
        return 424 * nodes + (layers - 1) * (nodes^2 + nodes) + 187 * nodes + 186
    end
    error("Unknown model: $modelname")
end

function format_nn_parameters(nparameters)
    if nparameters >= 1_000_000
        return "$(round(Int, nparameters / 1_000_000))M"
    end
    return "$(round(Int, nparameters / 1_000))k"
end

function format_count(count)
    count < 1_000 && return string(round(Int, count))
    count >= 1_000_000_000 && return "$(round(Int, count / 1_000_000_000))B"
    count >= 1_000_000 && return "$(round(Int, count / 1_000_000))M"
    return "$(round(Int, count / 1_000))k"
end

format_number(value) = replace(@sprintf("%.3g", value), "e" => "E")
format_residual(value) = @sprintf("%.1E", value)
function format_runtime(value)
    iszero(value) && return "0.0"
    if abs(value) < 1
        return replace(@sprintf("%.1g", value), "e" => "E")
    end
    return @sprintf("%.1f", value)
end
format_speedup(value) = isnan(value) ? "--" : format_runtime(value)

function solver_label(LinearSolver)
    solver = string(LinearSolver)
    occursin("Ma57Solver", solver) && return "MA57"
    occursin("Ma86Solver", solver) && return "MA86"
    occursin("Ma97Solver", solver) && return "MA97"
    occursin("SchurComplementSolver", solver) && return "Ours"
    error("Unknown linear solver: $solver")
end

function runtime_latex_contents(summary::DataFrames.DataFrame)
    summary = copy(summary)
    summary.model_label = uppercase.(summary.modelname)
    summary.solver_label = solver_label.(summary.LinearSolver)
    model_order = Dict("mnist" => 1, "scopf" => 2, "lsv" => 3)
    solver_order = Dict("MA57" => 1, "MA86" => 1, "Ours" => 2)
    summary.model_order = [model_order[row.modelname] for row in DataFrames.eachrow(summary)]
    summary.solver_order = [solver_order[row.solver_label] for row in DataFrames.eachrow(summary)]
    DataFrames.sort!(summary, [:model_order, :solver_order, :nodes, :layers])

    headers = ["Model", "Solver", "NN param.", "Init.", "Fact.", "Solve", "Resid.", "Refin. iter.", "Speedup"]
    rows = Vector{Vector{String}}()
    group_breaks = Set{Int}()
    groups = DataFrames.groupby(summary, [:model_label, :solver_label]; sort = false)
    for (group_index, group) in enumerate(groups)
        nrows = DataFrames.nrow(group)
        for (row_index, row) in enumerate(DataFrames.eachrow(group))
            model = row_index == 1 ? "\\multirow{$nrows}{*}{$(row.model_label)}" : ""
            solver = row_index == 1 ? "\\multirow{$nrows}{*}{$(row.solver_label)}" : ""
            nparameters = format_nn_parameters(
                nn_parameter_count(row.modelname, row.nodes, row.layers),
            )
            push!(rows, [
                model,
                solver,
                nparameters,
                format_runtime(row.t_init),
                format_runtime(row.t_factorize),
                format_runtime(row.t_solve),
                format_residual(row.residual),
                format_runtime(row.refine_iter),
                format_speedup(row.speedup),
            ])
        end
        group_index < length(groups) && push!(group_breaks, length(rows))
    end
    widths = [maximum(length(row[column]) for row in vcat([headers], rows)) for column in eachindex(headers)]
    format_row(row) = join(rpad.(row, widths), " & ") * " \\\\"

    lines = [format_row(headers)]
    for (row_index, row) in enumerate(rows)
        push!(lines, format_row(row))
        row_index in group_breaks && push!(lines, "\\midrule")
    end
    return join(lines, "\n")
end

function write_runtime_latex(io::IO, summary::DataFrames.DataFrame)
    print(io, runtime_latex_contents(summary))
    return nothing
end

function write_runtime_latex_file(fpath, summary::DataFrames.DataFrame)
    open(fpath, "w") do io
        write_runtime_latex(io, summary)
    end
    return fpath
end

function nn_structure_latex_contents(structures::DataFrames.DataFrame)
    structures = copy(structures)
    model_order = Dict("MNIST" => 1, "SCOPF" => 2, "LSV" => 3)
    structures.model_order = [model_order[row.model] for row in DataFrames.eachrow(structures)]
    DataFrames.sort!(structures, [:model_order, :layer_width, :layers])

    headers = ["Model", "N. inputs", "N. outputs", "Width", "N. layers", "N. param.", "Activation"]
    rows = Vector{Vector{String}}()
    group_breaks = Set{Int}()
    groups = DataFrames.groupby(structures, :model; sort = false)
    for (group_index, group) in enumerate(groups)
        nrows = DataFrames.nrow(group)
        for (row_index, row) in enumerate(DataFrames.eachrow(group))
            repeated = row_index == 1
            push!(rows, [
                repeated ? "\\multirow{$nrows}{*}{$(row.model)}" : "",
                repeated ? "\\multirow{$nrows}{*}{$(row.inputs)}" : "",
                repeated ? "\\multirow{$nrows}{*}{$(row.outputs)}" : "",
                string(row.layer_width),
                string(row.layers),
                format_nn_parameters(row.trained_parameters),
                row.activations,
            ])
        end
        group_index < length(groups) && push!(group_breaks, length(rows))
    end
    widths = [maximum(length(row[column]) for row in vcat([headers], rows)) for column in eachindex(headers)]
    format_row(row) = join(rpad.(row, widths), " & ") * " \\\\"

    lines = [format_row(headers), "\\midrule"]
    for (row_index, row) in enumerate(rows)
        push!(lines, format_row(row))
        row_index in group_breaks && push!(lines, "\\midrule")
    end
    return join(lines, "\n")
end

function write_nn_structure_latex_file(fpath, structures::DataFrames.DataFrame)
    open(fpath, "w") do io
        print(io, nn_structure_latex_contents(structures))
    end
    return fpath
end

function problem_structure_latex_contents(structures::DataFrames.DataFrame)
    structures = copy(structures)
    model_order = Dict("MNIST" => 1, "SCOPF" => 2, "LSV" => 3)
    structures.model_order = [model_order[row.model] for row in DataFrames.eachrow(structures)]
    DataFrames.sort!(structures, [:model_order, :trained_parameters])

    headers = ["Model", "N. param.", "N. var.", "N. con.", "Jac. NNZ", "Hess. NNZ"]
    rows = Vector{Vector{String}}()
    group_breaks = Set{Int}()
    groups = DataFrames.groupby(structures, :model; sort = false)
    for (group_index, group) in enumerate(groups)
        nrows = DataFrames.nrow(group)
        for (row_index, row) in enumerate(DataFrames.eachrow(group))
            push!(rows, [
                row_index == 1 ? "\\multirow{$nrows}{*}{$(row.model)}" : "",
                format_nn_parameters(row.trained_parameters),
                format_count(row.nvar),
                format_count(row.ncon),
                format_count(row.nnzj),
                format_count(row.nnzh),
            ])
        end
        group_index < length(groups) && push!(group_breaks, length(rows))
    end
    widths = [maximum(length(row[column]) for row in vcat([headers], rows)) for column in eachindex(headers)]
    format_row(row) = join(rpad.(row, widths), " & ") * " \\\\"

    lines = [format_row(headers), "\\midrule"]
    for (row_index, row) in enumerate(rows)
        push!(lines, format_row(row))
        row_index in group_breaks && push!(lines, "\\midrule")
    end
    return join(lines, "\n")
end

function write_problem_structure_latex_file(fpath, structures::DataFrames.DataFrame)
    open(fpath, "w") do io
        print(io, problem_structure_latex_contents(structures))
    end
    return fpath
end

function matrix_structure_latex_contents(structures::DataFrames.DataFrame)
    structures = copy(structures)
    structures.model_label = uppercase.(structures.modelname)
    matrix_labels = Dict(
        "Original KKT" => "KKT",
        "A" => "\$A\$",
        "B" => "\$B\$",
        "Pivot" => "Pivot matrix (\$C\$)",
        "Schur" => "Schur complement (\$S\$)",
    )
    structures.matrix_label = [matrix_labels[row.matrix_type] for row in DataFrames.eachrow(structures)]
    model_order = Dict("mnist" => 1, "scopf" => 2, "lsv" => 3)
    matrix_order = Dict("Original KKT" => 1, "A" => 2, "B" => 3, "Pivot" => 4, "Schur" => 5)
    structures.model_order = [model_order[row.modelname] for row in DataFrames.eachrow(structures)]
    structures.matrix_order = [matrix_order[row.matrix_type] for row in DataFrames.eachrow(structures)]
    DataFrames.sort!(structures, [:model_order, :nodes, :layers, :matrix_order])

    headers = ["Model", "NN param.", "Matrix", "N. row", "N. col", "NNZ"]
    rows = Vector{Vector{String}}()
    group_breaks = Set{Int}()
    groups = DataFrames.groupby(structures, [:model_label, :nodes, :layers]; sort = false)
    for (group_index, group) in enumerate(groups)
        nrows = DataFrames.nrow(group)
        for (row_index, row) in enumerate(DataFrames.eachrow(group))
            repeated = row_index == 1
            push!(rows, [
                repeated ? "\\multirow{$nrows}{*}{$(row.model_label)}" : "",
                repeated ? "\\multirow{$nrows}{*}{$(format_nn_parameters(nn_parameter_count(row.modelname, row.nodes, row.layers)))}" : "",
                row.matrix_label,
                format_count(row.nrow),
                format_count(row.ncol),
                format_count(row.nnz),
            ])
        end
        group_index < length(groups) && push!(group_breaks, length(rows))
    end
    widths = [maximum(length(row[column]) for row in vcat([headers], rows)) for column in eachindex(headers)]
    format_row(row) = join(rpad.(row, widths), " & ") * " \\\\"

    lines = [format_row(headers), "\\midrule"]
    for (row_index, row) in enumerate(rows)
        push!(lines, format_row(row))
        row_index in group_breaks && push!(lines, "\\midrule")
    end
    return join(lines, "\n")
end

function write_matrix_structure_latex_file(fpath, structures::DataFrames.DataFrame)
    open(fpath, "w") do io
        print(io, matrix_structure_latex_contents(structures))
    end
    return fpath
end

function compare_hsl_latex_contents(results::DataFrames.DataFrame)
    results = copy(results)
    results.solver_label = solver_label.(results.LinearSolver)
    solver_labels = ("MA57", "MA86", "MA97")
    model_order = ("mnist", "scopf", "lsv")
    lookup = Dict(
        (row.modelname, row.nodes, row.layers, row.solver_label) => row
        for row in DataFrames.eachrow(results)
    )

    headers = ["Model", "Solver", "NN param.", "Initialize", "Factorize", "Backsolve", "Residual"]
    rows = Vector{Vector{String}}()
    solver_breaks = Set{Int}()
    model_breaks = Set{Int}()
    format_compare_runtime(value) = value < 0.01 ? "\$< 0.01\$" : format_runtime(value)
    for (model_index, modelname) in enumerate(model_order)
        model_results = results[results.modelname .== modelname, :]
        nn_pairs = unique([(row.nodes, row.layers) for row in DataFrames.eachrow(model_results)])
        sort!(nn_pairs)
        nrows = length(solver_labels) * length(nn_pairs)
        for (solver_index, solver) in enumerate(solver_labels)
            for (nn_index, (nodes, layers)) in enumerate(nn_pairs)
                row = get(lookup, (modelname, nodes, layers, solver), nothing)
                nparameters = format_nn_parameters(nn_parameter_count(modelname, nodes, layers))
                if row === nothing
                    push!(rows, [
                        solver_index == 1 && nn_index == 1 ? "\\multirow{$nrows}[3]{*}{$(uppercase(modelname))}" : "",
                        nn_index == 1 ? "\\multirow{$(length(nn_pairs))}{*}{$solver}" : "",
                        nparameters * "*",
                        "--", "--", "--", "--",
                    ])
                else
                    push!(rows, [
                        solver_index == 1 && nn_index == 1 ? "\\multirow{$nrows}[3]{*}{$(uppercase(modelname))}" : "",
                        nn_index == 1 ? "\\multirow{$(length(nn_pairs))}{*}{$solver}" : "",
                        nparameters,
                        format_compare_runtime(row.t_init),
                        format_compare_runtime(row.t_factorize),
                        format_compare_runtime(row.t_solve),
                        format_residual(row.residual),
                    ])
                end
            end
            solver_index < length(solver_labels) && push!(solver_breaks, length(rows))
        end
        model_index < length(model_order) && push!(model_breaks, length(rows))
    end

    widths = [maximum(length(row[column]) for row in vcat([headers], rows)) for column in eachindex(headers)]
    format_row(row) = join(rpad.(row, widths), " & ") * " \\\\"
    lines = [format_row(headers), "\\midrule"]
    for (row_index, row) in enumerate(rows)
        push!(lines, format_row(row))
        if row_index in solver_breaks
            push!(lines, rpad("", widths[1]) * " \\cmidrule(ll){2-7}")
        elseif row_index in model_breaks
            push!(lines, "\\midrule")
        end
    end
    return join(lines, "\n")
end

function write_compare_hsl_latex_file(fpath, results::DataFrames.DataFrame)
    open(fpath, "w") do io
        print(io, compare_hsl_latex_contents(results))
    end
    return fpath
end

function flops_latex_contents(results::DataFrames.DataFrame)
    results = copy(results)
    results.model_label = uppercase.(results.modelname)
    results.solver_label = solver_label.(results.HSLLinearSolver)
    matrix_labels = Dict("kkt" => "KKT", "pivot" => "Pivot", "schur" => "Schur")
    results.matrix_label = [matrix_labels[row.matrix_type] for row in DataFrames.eachrow(results)]
    model_order = Dict("mnist" => 1, "scopf" => 2, "lsv" => 3)
    matrix_order = Dict("KKT" => 1, "Pivot" => 2, "Schur" => 3)
    results.model_order = [model_order[row.modelname] for row in DataFrames.eachrow(results)]
    results.matrix_order = [matrix_order[row.matrix_label] for row in DataFrames.eachrow(results)]
    DataFrames.sort!(results, [:model_order, :nodes, :layers, :matrix_order])

    headers = ["Model", "Solver", "NN param.", "Matrix", "Matrix dim.", "NNZ", "Factor NNZ", "FLOPs", "N. 2\$\\times\$2 pivots"]
    rows = Vector{Vector{String}}()
    group_breaks = Set{Int}()
    groups = DataFrames.groupby(results, [:model_label, :solver_label, :nodes, :layers]; sort = false)
    for (group_index, group) in enumerate(groups)
        nrows = DataFrames.nrow(group)
        for (row_index, row) in enumerate(DataFrames.eachrow(group))
            repeated = row_index == 1
            push!(rows, [
                repeated ? "\\multirow{$nrows}{*}{$(row.model_label)}" : "",
                repeated ? "\\multirow{$nrows}{*}{$(row.solver_label)}" : "",
                repeated ? "\\multirow{$nrows}{*}{$(format_nn_parameters(nn_parameter_count(row.modelname, row.nodes, row.layers)))}" : "",
                row.matrix_label,
                format_count(row.dim),
                format_count(row.nnz),
                format_count(row.factor_nnz),
                format_count(row.flops),
                format_count(row.n2by2),
            ])
        end
        group_index < length(groups) && push!(group_breaks, length(rows))
    end
    widths = [maximum(length(row[column]) for row in vcat([headers], rows)) for column in eachindex(headers)]
    format_row(row) = join(rpad.(row, widths), " & ") * " \\\\"

    lines = [format_row(headers), "\\midrule"]
    for (row_index, row) in enumerate(rows)
        push!(lines, format_row(row))
        row_index in group_breaks && push!(lines, "\\midrule")
    end
    return join(lines, "\n")
end

function write_flops_latex_file(fpath, results::DataFrames.DataFrame)
    open(fpath, "w") do io
        print(io, flops_latex_contents(results))
    end
    return fpath
end

function profile_schur_latex_contents(results::DataFrames.DataFrame)
    results = copy(results)
    results.model_label = uppercase.(results.modelname)
    model_order = Dict("mnist" => 1, "scopf" => 2, "lsv" => 3)
    results.model_order = [model_order[row.modelname] for row in DataFrames.eachrow(results)]
    DataFrames.sort!(results, [:model_order, :nodes, :layers])

    headers = ["", "", "", "Build Schur", "Schur", "Pivot", ""]
    rows = Vector{Vector{String}}()
    group_breaks = Set{Int}()
    groups = DataFrames.groupby(results, :model_label; sort = false)
    for (group_index, group) in enumerate(groups)
        nrows = DataFrames.nrow(group)
        for (row_index, row) in enumerate(DataFrames.eachrow(group))
            factorize_percentage = 100 .* (
                row.construct_schur,
                row.factorize_schur,
                row.factorize_pivot,
            ) ./ row.t_factorize
            push!(rows, [
                row_index == 1 ? "\\multirow{$nrows}{*}{$(row.model_label)}" : "",
                format_nn_parameters(nn_parameter_count(row.modelname, row.nodes, row.layers)),
                format_runtime(row.t_factorize),
                format_count(factorize_percentage[1]),
                format_count(factorize_percentage[2]),
                format_count(factorize_percentage[3]),
                format_runtime(row.t_solve),
            ])
        end
        group_index < length(groups) && push!(group_breaks, length(rows))
    end
    widths = [maximum(length(row[column]) for row in vcat([headers], rows)) for column in eachindex(headers)]
    format_row(row) = join(rpad.(row, widths), " & ") * " \\\\"
#=
    title_row = "\\multirow{2}{*}{Model} & \\\multirow{2}{*}{NN param.} & \\\multirow{2}{*}{Factorize (s)} & \\\multicolumn{3}{c}{Percent of factorize time (\\%)} & \\\multirow{2}{*}{Backsolve (s)} \\\\\\\"
=#
    title_row = raw"\multirow{2}{*}{Model} & \multirow{2}{*}{NN param.} & \multirow{2}{*}{Factorize (s)} & \multicolumn{3}{c}{Percent of factorize time (\%)} & \multirow{2}{*}{Backsolve (s)} \\"
    lines = [title_row, "\\cmidrule{4-6}", format_row(headers), "\\midrule"]
    for (row_index, row) in enumerate(rows)
        push!(lines, format_row(row))
        row_index in group_breaks && push!(lines, "\\midrule")
    end
    return join(lines, "\n")
end

function write_profile_schur_latex_file(fpath, results::DataFrames.DataFrame)
    open(fpath, "w") do io
        print(io, profile_schur_latex_contents(results))
    end
    return fpath
end

function summarize_runtime_results(results)
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
    is_baseline(row) = occursin(r"MadNLPHSL\.Ma(57|86)Solver", string(row.LinearSolver))
    baseline_times = Dict(
        key(row) => (row.t_factorize + row.t_solve) / row.n_iterates
        for row in DataFrames.eachrow(summary) if is_baseline(row)
    )
    summary.speedup = [
        occursin("SchurComplementSolver", string(row.LinearSolver)) ?
        get(baseline_times, key(row), NaN) / ((row.t_factorize + row.t_solve) / row.n_iterates) :
        NaN
        for row in DataFrames.eachrow(summary)
    ]
    return summary
end

function read_runtime_results()
    results_dir = joinpath(@__DIR__, "results")
    first_results = CSV.read(joinpath(results_dir, "runtime-first.csv"), DataFrames.DataFrame)
    last_results = CSV.read(joinpath(results_dir, "runtime-last.csv"), DataFrames.DataFrame)
    return (; first_results, last_results, combined_results = vcat(first_results, last_results))
end

function add_iteration_labels(results; last = false)
    results = copy(results)
    results.iteration_label = fill("", DataFrames.nrow(results))
    results.iteration_order = zeros(Int, DataFrames.nrow(results))
    results.iterate_set_order = fill(last ? 2 : 1, DataFrames.nrow(results))
    group_columns = [:modelname, :nodes, :layers, :LinearSolver]
    for group in DataFrames.groupby(results, group_columns; sort = false)
        nrows = DataFrames.nrow(group)
        for (i, row) in enumerate(DataFrames.eachrow(group))
            row.iteration_label = last ? (i == nrows ? "N" : "N-$(nrows - i)") : string(i)
            row.iteration_order = i
        end
    end
    return results
end

function complete_runtime_latex_contents(first_results, last_results)
    results = vcat(
        add_iteration_labels(first_results),
        add_iteration_labels(last_results; last = true),
    )
    results.model_label = uppercase.(results.modelname)
    results.solver_label = solver_label.(results.LinearSolver)
    baseline_times = Dict(
        (row.modelname, row.nodes, row.layers, row.iterate_set_order, row.iteration_order) =>
        row.t_factorize + row.t_solve
        for row in DataFrames.eachrow(results)
        if occursin(r"MadNLPHSL\.Ma(57|86)Solver", string(row.LinearSolver))
    )
    results.speedup = [
        occursin("SchurComplementSolver", string(row.LinearSolver)) ?
        get(
            baseline_times,
            (row.modelname, row.nodes, row.layers, row.iterate_set_order, row.iteration_order),
            NaN,
        ) / (row.t_factorize + row.t_solve) :
        NaN
        for row in DataFrames.eachrow(results)
    ]
    model_order = Dict("mnist" => 1, "scopf" => 2, "lsv" => 3)
    solver_order = Dict("MA57" => 1, "MA86" => 1, "Ours" => 2)
    results.model_order = [model_order[row.modelname] for row in DataFrames.eachrow(results)]
    results.solver_order = [solver_order[row.solver_label] for row in DataFrames.eachrow(results)]
    DataFrames.sort!(results, [
        :model_order, :solver_order, :nodes, :layers, :iterate_set_order, :iteration_order,
    ])

    headers = ["Model", "Solver", "NN", "Iter.", "Init.", "Fact.", "Solve", "Neg. eig.", "Residual", "Refin. iter.", "Speedup"]
    rows = Vector{Vector{String}}()
    group_breaks = Set{Int}()
    groups = DataFrames.groupby(results, [:model_label, :solver_label, :nodes, :layers]; sort = false)
    for (group_index, group) in enumerate(groups)
        for row in DataFrames.eachrow(group)
            push!(rows, [
                row.model_label,
                row.solver_label,
                format_nn_parameters(nn_parameter_count(row.modelname, row.nodes, row.layers)),
                row.iteration_label,
                format_runtime(row.t_init),
                format_runtime(row.t_factorize),
                format_runtime(row.t_solve),
                string(row.nneg_eig),
                format_residual(row.residual),
                string(round(Int, row.refine_iter)),
                format_speedup(row.speedup),
            ])
        end
        group_index < length(groups) && push!(group_breaks, length(rows))
    end
    widths = [maximum(length(row[column]) for row in vcat([headers], rows)) for column in eachindex(headers)]
    format_row(row) = join(rpad.(row, widths), " & ") * " \\\\"
    runtime_headers = ["", "", "", "", "Init.", "Fact.", "Solve", "", "", "", ""]

    lines = [
        "\\multirow{2}{*}{Model} & \\multirow{2}{*}{Solver} & \\multirow{2}{*}{NN} & \\multirow{2}{*}{Iter.} & \\multicolumn{3}{c}{Runtime (s)} & \\multirow{2}{*}{Neg. eig.} & \\multirow{2}{*}{Residual} & \\multirow{2}{*}{Refinement iter.} & \\multirow{2}{*}{Speedup}\\\\",
        "\\cmidrule{5-7}",
        format_row(runtime_headers),
        "\\midrule",
        "\\endhead",
    ]
    for (row_index, row) in enumerate(rows)
        push!(lines, format_row(row))
        row_index in group_breaks && push!(lines, "\\midrule")
    end
    return join(lines, "\n")
end

function parse_latex_commandline()
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
        "experiment"
            help = "Experiment to format"
            required = true
    end
    return ArgParse.parse_args(settings)
end

function write_latex_main()
    args = parse_latex_commandline()
    if args["experiment"] == "runtime"
        results = read_runtime_results()
        summaries = map(summarize_runtime_results, results)
        results_dir = joinpath(@__DIR__, "results")
        filenames = (
            "runtime-first-summary.txt",
            "runtime-last-summary.txt",
            "runtime-both-summary.txt",
        )
        written_files = String[]
        for (label, summary, filename) in zip(keys(summaries), values(summaries), filenames)
            push!(written_files, write_runtime_latex_file(joinpath(results_dir, filename), summary))
            println("% $(replace(string(label), "_" => " "))")
            write_runtime_latex(stdout, summary)
            println("\n")
        end
        complete_path = joinpath(results_dir, "complete-runtime.txt")
        open(complete_path, "w") do io
            print(io, complete_runtime_latex_contents(results.first_results, results.last_results))
        end
        push!(written_files, complete_path)
        println("Wrote:")
        println.(written_files)
        return summaries
    elseif args["experiment"] == "nn-structure"
        results_dir = joinpath(@__DIR__, "results")
        structures = CSV.read(joinpath(results_dir, "nn-structure.csv"), DataFrames.DataFrame)
        fpath = write_nn_structure_latex_file(joinpath(results_dir, "nn-structure.txt"), structures)
        print(nn_structure_latex_contents(structures))
        println("\nWrote $fpath")
        return structures
    elseif args["experiment"] == "problem-structure"
        results_dir = joinpath(@__DIR__, "results")
        structures = CSV.read(joinpath(results_dir, "problem-structure.csv"), DataFrames.DataFrame)
        fpath = write_problem_structure_latex_file(joinpath(results_dir, "problem-structure.txt"), structures)
        print(problem_structure_latex_contents(structures))
        println("\nWrote $fpath")
        return structures
    elseif args["experiment"] == "flops"
        results_dir = joinpath(@__DIR__, "results")
        results = CSV.read(joinpath(results_dir, "hsl-flops.csv"), DataFrames.DataFrame)
        fpath = write_flops_latex_file(joinpath(results_dir, "hsl-flops.txt"), results)
        print(flops_latex_contents(results))
        println("\nWrote $fpath")
        return results
    elseif args["experiment"] == "profile-schur"
        results_dir = joinpath(@__DIR__, "results")
        results = CSV.read(joinpath(results_dir, "profile-schur.csv"), DataFrames.DataFrame)
        fpath = write_profile_schur_latex_file(joinpath(results_dir, "profile-schur.txt"), results)
        print(profile_schur_latex_contents(results))
        println("\nWrote $fpath")
        return results
    elseif args["experiment"] == "compare-hsl"
        results_dir = joinpath(@__DIR__, "results")
        results = CSV.read(joinpath(results_dir, "compare-hsl.csv"), DataFrames.DataFrame)
        fpath = write_compare_hsl_latex_file(joinpath(results_dir, "compare-hsl.txt"), results)
        print(compare_hsl_latex_contents(results))
        println("\nWrote $fpath")
        return results
    end
    error("Unknown experiment: $(args["experiment"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_latex_main()
end
