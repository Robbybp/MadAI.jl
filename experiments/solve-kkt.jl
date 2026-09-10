import JSON
import SparseArrays
import LinearAlgebra
import Statistics
import JuMP
import NLPModelsJuMP
import MadNLP
import MadNLPHSL
import MadAI

function MadNLP.solve!(solver::MadNLP.AbstractLinearSolver, rhs::Vector)
    return MadNLP.solve_linear_system!(solver, rhs)
end

function _hsl_factorization_data(solver::MadNLPHSL.Ma57Solver)
    return (; factor_size = solver.info[14], flops = solver.rinfo[3] + solver.rinfo[4],
        n2by2 = solver.info[22], status_code = solver.info[1])
end

function _hsl_factorization_data(solver::MadNLPHSL.Ma86Solver)
    return (; factor_size = solver.info.num_factor, flops = solver.info.num_flops,
        n2by2 = solver.info.num_two, status_code = solver.info.flag)
end

function _schur_rhs(solver::MadAI.SchurComplementSolver, rhs)
    dim = solver.csc.n
    pivot_indices = solver.pivot_indices
    pivot_index_set = Set(pivot_indices)
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:dim)
    B = solver.csc[pivot_indices, reduced_indices] + solver.csc[reduced_indices, pivot_indices]'

    pivot_rhs = copy(rhs[pivot_indices])
    MadNLP.solve!(solver.pivot_solver, pivot_rhs)
    return rhs[reduced_indices] - B' * pivot_rhs
end

function _pivot_rhs(solver::MadAI.SchurComplementSolver, rhs)
    dim = solver.csc.n
    pivot_indices = solver.pivot_indices
    pivot_index_set = Set(pivot_indices)
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:dim)
    B = solver.csc[pivot_indices, reduced_indices] + solver.csc[reduced_indices, pivot_indices]'

    reduced_solution = _schur_rhs(solver, rhs)
    MadNLP.solve!(solver.reduced_solver, reduced_solution)
    return rhs[pivot_indices] - B * reduced_solution
end

function _factorize_and_refine!(solver, matrix, rhs)
    solver.csc.nzval .= matrix.nzval
    _t = time()
    MadNLP.factorize!(solver)
    t_factorize = time() - _t

    solution = copy(rhs)
    full_matrix, tril_to_full_view = MadNLP.get_tril_to_full(matrix)
    _t = time()
    MadNLP.solve!(solver, solution)
    refine_result = MadAI.refine!(solution, solver, rhs;
        max_iter = 64, tol = 1e-8, full_matrix, tril_to_full_view)
    t_solve = time() - _t
    residual = maximum(abs.(full_matrix * solution - rhs))

    return merge((;
        dim = matrix.m,
        nnz = SparseArrays.nnz(matrix),
        t_factorize,
        t_solve,
        residual,
        refine_success = refine_result.success,
        refine_iter = refine_result.iterations,
    ), _hsl_factorization_data(solver))
end

function benchmark_kkt_matrices(
    nlp,
    HSLLinearSolver::Type{<:MadNLP.AbstractLinearSolver},
    hsl_options,
    schur_options,
    iterates;
    kwds...,
)
    madnlp = MadNLP.MadNLPSolver(nlp; linear_solver = HSLLinearSolver, kwds...)
    MadNLP.initialize!(madnlp)
    kkt_matrix = MadNLP.get_kkt(MadNLP.get_kkt(madnlp))
    schur_solver = MadAI.SchurComplementSolver(kkt_matrix; opt = schur_options)
    hsl_solvers = Dict{Symbol,Any}()
    results = NamedTuple[]

    for (i, iterate) in enumerate(iterates)
        println("BENCHMARKING KKT MATRICES FOR ITERATE $i")
        matrix, rhs = iterate_to_kkt(madnlp, Dict(iterate))
        schur_solver.csc.nzval .= matrix.nzval
        MadNLP.factorize!(schur_solver)

        pivot_matrix = schur_solver.pivot_solver.csc
        schur_matrix = schur_solver.reduced_solver.csc
        matrix_rhs = (
            kkt = (matrix, rhs),
            pivot = (pivot_matrix, _pivot_rhs(schur_solver, rhs)),
            schur = (schur_matrix, _schur_rhs(schur_solver, rhs)),
        )
        for (matrix_type, (submatrix, subrhs)) in pairs(matrix_rhs)
            solver = get!(hsl_solvers, matrix_type) do
                HSLLinearSolver(submatrix; opt = hsl_options)
            end
            push!(results, merge((; iterate = i, matrix_type),
                _factorize_and_refine!(solver, submatrix, subrhs)))
        end
    end
    return results
end

function iterate_to_kkt(nlp::NLPModelsJuMP.MathOptNLPModel, iterate::Dict)
    madnlp = MadNLP.MadNLPSolver(nlp)
    MadNLP.initialize!(madnlp)
    return iterate_to_kkt(madnlp, iterate)
end

function get_matrices_structure(
    nlp,
    opt_linear_solver;
    MadNLPLinearSolver::Type{<:MadNLP.AbstractLinearSolver} = MadNLPHSL.Ma57Solver,
    kwds...,
)
    madnlp = MadNLP.MadNLPSolver(nlp; linear_solver = MadNLPLinearSolver, kwds...)
    MadNLP.initialize!(madnlp)
    kkt_matrix = MadNLP.get_kkt(MadNLP.get_kkt(madnlp))
    schur_solver = MadAI.SchurComplementSolver(kkt_matrix; opt = opt_linear_solver)

    pivot_indices = schur_solver.pivot_indices
    pivot_index_set = Set(pivot_indices)
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:kkt_matrix.n)
    A = kkt_matrix[reduced_indices, reduced_indices]
    B = kkt_matrix[pivot_indices, reduced_indices] +
        kkt_matrix[reduced_indices, pivot_indices]'

    matrices = (
        ("Original KKT", kkt_matrix),
        ("A", A),
        ("B", B),
        ("Pivot", schur_solver.pivot_solver.csc),
        ("Schur", schur_solver.reduced_solver.csc),
    )
    return [
        (; matrix_type, nrow = Int(matrix.m), ncol = Int(matrix.n), nnz = SparseArrays.nnz(matrix))
        for (matrix_type, matrix) in matrices
    ]
end

function iterate_to_kkt(madnlp::MadNLP.MadNLPSolver, iterate::Dict)
    x = iterate["primal"]
    y = iterate["dual"]
    zL = iterate["Ldual"]
    zU = iterate["Udual"]
    μ = iterate["barrier"]

    # TODO: Make sure this is right
    # Values must use MadNLP's *reformulated* ordering.
    MadNLP.full(MadNLP.get_x(madnlp)) .= x
    MadNLP.get_y(madnlp) .= y
    # TODO: Fix dimension mismatch errors
    MadNLP.full(MadNLP.get_zl(madnlp)) .= zL
    MadNLP.full(MadNLP.get_zu(madnlp)) .= zU
    MadNLP.set_mu!(madnlp, μ)

    # Re-evaluate quantities that depend on x and y.
    MadNLP.set_obj_val!(madnlp,
        MadNLP.eval_f_wrapper(madnlp, MadNLP.get_x(madnlp)))
    MadNLP.eval_cons_wrapper!(madnlp, MadNLP.get_c(madnlp), MadNLP.get_x(madnlp))
    MadNLP.eval_grad_f_wrapper!(madnlp, MadNLP.get_f(madnlp), MadNLP.get_x(madnlp))
    MadNLP.eval_jac_wrapper!(madnlp, MadNLP.get_kkt(madnlp), MadNLP.get_x(madnlp))
    MadNLP.eval_lag_hess_wrapper!(
        madnlp, MadNLP.get_kkt(madnlp), MadNLP.get_x(madnlp), MadNLP.get_y(madnlp),
    )

    # Form the augmented KKT system and its RHS.
    MadNLP.set_aug_diagonal!(MadNLP.get_kkt(madnlp), madnlp)
    MadNLP.jtprod!(MadNLP.get_jacl(madnlp), MadNLP.get_kkt(madnlp), MadNLP.get_y(madnlp))
    MadNLP.set_aug_rhs!(
        madnlp, MadNLP.get_kkt(madnlp), MadNLP.get_c(madnlp), MadNLP.get_mu(madnlp),
    )
    MadNLP.dual_inf_perturbation!(
        MadNLP.primal(MadNLP.get_p(madnlp)),
        MadNLP.get_ind_llb(madnlp),
        MadNLP.get_ind_uub(madnlp),
        MadNLP.get_mu(madnlp),
        MadNLP.get_opt(madnlp).kappa_d,
    )

    madnlp_matrix = MadNLP.get_kkt(MadNLP.get_kkt(madnlp))
    if haskey(iterate, "regularized_kkt_diagonal")
        regularized_diagonal = iterate["regularized_kkt_diagonal"]
        @assert length(regularized_diagonal) == size(madnlp_matrix, 1)
        madnlp_matrix[LinearAlgebra.diagind(madnlp_matrix)] .= regularized_diagonal
    end
    rhs = copy(MadNLP.primal_dual(MadNLP.get_p(madnlp)))
    return madnlp_matrix, rhs
end

function solve_kkt(
    nlp,
    LinearSolver,
    opt_linear_solver,
    iterates;
    MadNLPLinearSolver::Type{<:MadNLP.AbstractLinearSolver} = MadNLPHSL.Ma57Solver,
    kwds...,
)
    # NOTE: MadNLP is initialized with a linear solver, but this solver is only
    # used to construct the KKT matrix.
    # The worst thing that happens here is that the linear solver has long
    # initialization time. I think MA57 has good initialization time by default
    # on all instances.
    println("INITIALIZING MADNLP")
    _t = time()
    madnlp = MadNLP.MadNLPSolver(nlp; linear_solver = MadNLPLinearSolver, kwds...)
    MadNLP.initialize!(madnlp)
    println("TIME TO INITIALIZE MADNLP: $(time() - _t)")

    matrix = MadNLP.get_kkt(MadNLP.get_kkt(madnlp))
    _t = time()
    linear_solver = LinearSolver(matrix; opt = opt_linear_solver)
    full_matrix, tril_to_full_view = MadNLP.get_tril_to_full(matrix)
    t_init = time() - _t

    results = Any[]
    for (i, iterate) in enumerate(iterates)
        println("SOLVING KKT FOR ITERATE $i")
        iterate = Dict(iterate)
        madnlp_matrix, rhs = iterate_to_kkt(madnlp, iterate)

        sol = copy(rhs)
        # TODO: Construct derived matrix if necessary
        linear_solver.csc.nzval .= madnlp_matrix.nzval

        local _t = time()
        MadNLP.factorize!(linear_solver)
        npos, nzero, nneg = MadNLP.inertia(linear_solver)
        t_factorize = time() - _t

        _t = time()
        MadNLP.solve!(linear_solver, sol)
        # Confusingly, the full matrix gets updated using the linear solver's matrix
        # as part of this function, so there is no need to update it beforehand.
        # But here's how it would be done if we needed to:
        #     full_matrix.nzval .= tril_to_full_view
        refine_res = MadAI.refine!(
            sol,
            linear_solver,
            rhs;
            max_iter = 64,
            tol = 1e-8,
            full_matrix,
            tril_to_full_view,
        )
        t_solve = time() - _t

        residual = maximum(abs.(full_matrix * sol - rhs))

        push!(results,
            (;
                dim = matrix.m,
                nnz = SparseArrays.nnz(matrix),
                # Other identifies, like model and solver, will be added one level up
                t_init,
                t_factorize,
                t_solve,
                nneg_eig = nneg,
                residual,
                refine_success = refine_res.success,
                refine_iter = refine_res.iterations,
            )
        )
    end
    return results
end

function profile_schur(
    nlp,
    opt_linear_solver,
    iterates;
    MadNLPLinearSolver::Type{<:MadNLP.AbstractLinearSolver} = MadNLPHSL.Ma57Solver,
    kwds...,
)
    madnlp = MadNLP.MadNLPSolver(nlp; linear_solver = MadNLPLinearSolver, kwds...)
    MadNLP.initialize!(madnlp)

    matrix = MadNLP.get_kkt(MadNLP.get_kkt(madnlp))
    linear_solver = MadAI.SchurComplementSolver(matrix; opt = opt_linear_solver)
    full_matrix, tril_to_full_view = MadNLP.get_tril_to_full(matrix)

    t_factorize = 0.0
    t_solve = 0.0
    t_resid = 0.0
    residuals = Float64[]
    refine_iterations = Int[]
    refine_success = true
    for (i, iterate) in enumerate(iterates)
        println("PROFILING SCHUR SOLVER FOR ITERATE $i")
        madnlp_matrix, rhs = iterate_to_kkt(madnlp, Dict(iterate))
        linear_solver.csc.nzval .= madnlp_matrix.nzval

        _t = time()
        MadNLP.factorize!(linear_solver)
        t_factorize += time() - _t

        sol = copy(rhs)
        _t = time()
        MadNLP.solve!(linear_solver, sol)
        refine_res = MadAI.refine!(
            sol,
            linear_solver,
            rhs;
            max_iter = 64,
            tol = 1e-8,
            full_matrix,
            tril_to_full_view,
        )
        t_solve += time() - _t
        t_resid += refine_res.t_resid
        push!(residuals, maximum(abs.(full_matrix * sol - rhs)))
        push!(refine_iterations, refine_res.iterations)
        refine_success &= refine_res.success
    end

    timer = linear_solver.timer
    factorize_schur = timer.factorize.reduced
    factorize_pivot = timer.factorize.pivot
    construct_schur = timer.factorize.solve + timer.factorize.multiply
    solve_schur = timer.solve_timer.solve_schur
    solve_pivot = timer.solve_timer.solve_pivot
    compute_rhs = timer.solve_timer.compute_rhs
    return (; n_iterates = length(iterates),
        dim = matrix.m,
        nnz = SparseArrays.nnz(matrix),
        t_factorize,
        factorize_schur,
        factorize_pivot,
        construct_schur,
        other_factorize = t_factorize - factorize_schur - factorize_pivot - construct_schur,
        t_solve,
        solve_schur,
        solve_pivot,
        compute_rhs,
        compute_resid = t_resid,
        other_backsolve = t_solve - t_resid - solve_schur - solve_pivot - compute_rhs,
        residual = Statistics.mean(residuals),
        refine_iter = Statistics.mean(refine_iterations),
        refine_success,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    include("JuMP/models.jl")

    modelname = "mnist"
    nodes = 128
    layers = 4

    datadir = joinpath(@__DIR__, "data", "iterates")
    fname = "$modelname-$(nodes)nodes$(layers)layers-first.json"
    fpath = joinpath(datadir, fname)
    iterate_data = open(fpath, "r") do io
        JSON.parse(io)
    end

    model, formulation = get_model(modelname, nodes, layers)
    use_ma57 = false
    if use_ma57
        LinearSolver = MadNLPHSL.Ma57Solver
        opt_linear_solver = MadNLP.default_options(LinearSolver)
    else
        LinearSolver = MadAI.SchurComplementSolver
        pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
        pivot_indices = MadAI.get_kkt_indices(model, pivot_vars, pivot_cons)
        pivot_indices = convert(Vector{Int32}, pivot_indices)
        blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
        pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
        opt_linear_solver = MadAI.SchurComplementOptions(;
            ReducedSolver = MadNLPHSL.Ma57Solver,
            PivotSolver = MadAI.BlockTriangularSolver,
            pivot_indices,
            pivot_solver_opt,
        )
    end

    variables, constraints = MadAI.get_var_con_order(model)
    @assert all(iterate_data["variables"] .== string.(JuMP.index.(variables)))
    @assert all(iterate_data["constraints"] .== string.(JuMP.index.(constraints)))
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    results = solve_kkt(nlp, LinearSolver, opt_linear_solver, iterate_data["iterates"])
    display(results)
end
