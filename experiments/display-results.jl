import CSV
import DataFrames

include("reproduce.jl")

function main()
    results_dir = joinpath(@__DIR__, "results")
    first_results = CSV.read(joinpath(results_dir, "runtime-first.csv"), DataFrames.DataFrame)
    last_results = CSV.read(joinpath(results_dir, "runtime-last.csv"), DataFrames.DataFrame)

    println("First iterates")
    println("--------------")
    display(summarize_results(first_results))
    println("Last iterates")
    println("-------------")
    display(summarize_results(last_results))
    println("Combined iterates")
    println("-----------------")
    display(summarize_results(vcat(first_results, last_results)))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
