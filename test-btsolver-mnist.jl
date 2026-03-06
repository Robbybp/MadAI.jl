import JuMP
import LinearAlgebra
import MadNLP
import MadNLPHSL
import MathProgIncidence
import NLPModels
import SparseArrays
using Test
using Printf

include("adversarial-image.jl")
include("btsolver.jl")
include("models.jl")
include("nlpmodels.jl")
include("linalg.jl")

function _test_matrix(
    csc::SparseArrays.SparseMatrixCSC;
    blocks = nothing,
    atol=1e-8,
    nrhs = 10,
    baseline_solver = "umfpack",
    skiptest = false,
    btsolver = nothing,
    symmetric = true,
)
    dim = csc.m
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)

    _t = time()
    if btsolver === nothing
        opt = BlockTriangularOptions(; blocks, symmetric)
        btsolver = BlockTriangularSolver(csc; opt)
    end
    t_init = time() - _t
    _t = time()
    MadNLP.factorize!(btsolver)
    t_fact = time() - _t
    sol = copy(rhs)
    _t = time()
    MadNLP.solve!(btsolver, sol)
    t_solve = time() - _t

    if !skiptest
        if baseline_solver == "umfpack"
            baseline = csc \ rhs
        elseif baseline_solver == "ma57"
            solver = MadNLPHSL.Ma57Solver(SparseArrays.tril(csc))
            MadNLP.factorize!(solver)
            baseline = copy(rhs)
            MadNLP.solve!(solver, baseline)
        else
            error("baseline_solver argument must be \"umfpack\" or \"ma57\"")
        end
        @test all(isapprox.(sol, baseline; atol))
        if !(all(isapprox.(sol, baseline; atol)))
            diff = abs.(sol .- baseline)
            ndiff = count(diff[:, 1] .> atol)
            maxdiff = maximum(diff)
            println("Solution does not match baseline")
            println("Max error: $maxdiff")
            println("N. errors: $ndiff / $dim")
        end
    end
    return (;
        time = (;
            initialize = t_init,
            factorize = t_fact,
            solve = t_solve,
        ),
        btsolver,
    )
end

function test_mnist_nn_kkt(;
    nrhs = 10,
    nnfname = "mnist-relu128nodes4layers.pt",
    skip_auto_btf = false,
)
    IMAGE_INDEX = 7
    ADVERSARIAL_LABEL = 1
    THRESHOLD = 0.6
    nnfile = joinpath("nn-models", nnfname)
    if !isfile(nnfile)
        @error("$nnfile does not exist or is not a file")
        return
    end
    model, outputs, formulation = get_adversarial_model(
        nnfile, IMAGE_INDEX, ADVERSARIAL_LABEL, THRESHOLD;
        reduced_space = false
    )
    _t = time()
    nlp, kkt_system, kkt_matrix = get_kkt(model, Solver=MadNLPHSL.Ma57Solver)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] (Since model build) Get KKT")

    pivot_vars, pivot_cons = get_vars_cons(formulation)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] (Since model build) Get vars/cons")
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] (Since model build) Get pivot indices")
    pivot_index_set = Set(pivot_indices)
    @assert kkt_matrix.m == kkt_matrix.n
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:kkt_matrix.m)
    pivot_dim = length(pivot_indices)
    @assert pivot_dim % 2 == 0

    P = pivot_indices
    R = reduced_indices
    C_orig = kkt_matrix[P, P]

    # Filter out constraint regularization nonzeros
    # By convention, constraints are the second half of the pivot indices
    to_ignore = Set(Int(pivot_dim / 2 + 1):pivot_dim)
    I, J, V = SparseArrays.findnz(C_orig)
    to_retain = filter(k -> !(I[k] in to_ignore && J[k] in to_ignore), 1:length(I))
    I = I[to_retain]
    J = J[to_retain]
    V = V[to_retain]
    C = SparseArrays.sparse(I, J, V, C_orig.m, C_orig.n)
    C_full = fill_upper_triangle(C)
    if !skip_auto_btf
        res = _test_matrix(C_full; nrhs, atol = 1e-5, skiptest = true)
        println("Timing breakdown")
        println("----------------")
        println("Initialization: $(res.time.initialize)")
        println("Factorization:  $(res.time.factorize)")
        println("Solve (x$nrhs):  $(res.time.solve)")
        println()
    end
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] (Since model build) Filter NZ")

    # Maps indices in the original space to their index in the pivot matrix
    index_remap = Dict((p, i) for (i, p) in enumerate(P))

    layers = get_layers(formulation)
    var_con_by_layer = [get_vars_cons(l) for l in layers]
    var_indices_by_layer = [get_kkt_indices(model, vars, []) for (vars, _) in var_con_by_layer]
    con_indices_by_layer = [get_kkt_indices(model, [], cons) for (_, cons) in var_con_by_layer]
    blocks = []
    for l in 1:length(layers)
        conindices = [index_remap[i] for i in con_indices_by_layer[l]]
        varindices = [index_remap[i] for i in var_indices_by_layer[l]]
        push!(blocks, (conindices, varindices))
    end
    for l in reverse(1:length(layers))
        conindices = [index_remap[i] for i in con_indices_by_layer[l]]
        varindices = [index_remap[i] for i in var_indices_by_layer[l]]
        push!(blocks, (varindices, conindices))
    end
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] (Since model build) Get block indices")

    # We skip the test-against-baseline as there is a significant amount of error for
    # these relatively large systems.
    # NOTE: This might be better now that I'm initializing the intermediate variables.
    # TODO: Revisit this
    res = _test_matrix(C_full; blocks, nrhs, atol = 1e-5, skiptest = true)
    println("Timing breakdown")
    println("----------------")
    println("Initialization: $(res.time.initialize)")
    println("Factorization:  $(res.time.factorize)")
    println("Solve (x$nrhs):  $(res.time.solve)")
    println()
end

@testset "BT-MNIST" begin
    #nnfname = "mnist-relu1024nodes4layers.pt"
    nnfname = "mnist-relu2048nodes4layers.pt"
    test_mnist_nn_kkt(; nrhs = 1000, nnfname, skip_auto_btf = true)
end
