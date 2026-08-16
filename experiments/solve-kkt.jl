import JSON
import SparseArrays
import JuMP
import NLPModelsJuMP
import MadNLP
import MadNLPHSL
import MadAI

function MadNLP.solve!(solver::MadNLP.AbstractLinearSolver, rhs::Vector)
    return MadNLP.solve_linear_system!(solver, rhs)
end

include("JuMP/models.jl")

modelname = "mnist"
nodes = 128
layers = 4

DATADIR = joinpath(@__DIR__, "data", "iterates")
fname = "$modelname-$(nodes)nodes$(layers)layers-first.json"
fpath = joinpath(DATADIR, fname)
iterate_data = open(fpath, "r") do io
    return JSON.parse(io)
end

USE_MA57 = false
if USE_MA57
    LinearSolver = MadNLPHSL.Ma57Solver
    opt = MadNLP.default_options(LinearSolver)
else
    LinearSolver = MadAI.SchurComplementSolver
    pivot_vars, pivot_cons = MadAI.get_vars_cons(formulation)
    pivot_indices = MadAI.get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = MadAI.partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = MadAI.BlockTriangularOptions(; blocks)
    opt = MadAI.SchurComplementOptions(;
        ReducedSolver = MadNLPHSL.Ma57Solver,
        PivotSolver = MadAI.BlockTriangularSolver,
        pivot_indices,
        pivot_solver_opt,
    )
end

# The above are all inputs into this function

model, formulation = get_model(modelname, nodes, layers)
# These three lines stay with the model. If the model is an input, these
# will get popped up.
variables, constraints = MadAI.get_var_con_order(model)
@assert all(iterate_data["variables"] .== string.(JuMP.index.(variables)))
@assert all(iterate_data["constraints"] .== string.(JuMP.index.(constraints)))
nlp = NLPModelsJuMP.MathOptNLPModel(model)

# TODO: Options and print level
# NOTE: MadNLP is initialize with a linear solver, but this doesn't matter for
# our purposes. We will never use this linear solver for anything. We only use
# MadNLP to construct the KKT matrix.
madnlp = MadNLP.MadNLPSolver(nlp)
MadNLP.initialize!(madnlp)

matrix = MadNLP.get_kkt(MadNLP.get_kkt(madnlp))
# TODO: Extract derived matrix if necessary.
# This matrix is used to initialize the linear solver. We must update it
# in-place every time we change to a new iterate.

_t = time()
linear_solver = LinearSolver(matrix; opt)
# This is necessary for iterative refinement.
full_matrix, tril_to_full_view = MadNLP.get_tril_to_full(matrix)
t_init = time() - _t

data = Any[]
for iterate in iterate_data["iterates"]
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
    rhs = MadNLP.primal_dual(MadNLP.get_p(madnlp))
    sol = copy(rhs)
    # TODO: Construct derived matrix if necessary
    linear_solver.csc .= madnlp_matrix

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
        max_iter = 20,
        tol = 1e-5,
        full_matrix,
        tril_to_full_view,
    )
    t_solve = time() - _t

    residual = maximum(abs.(full_matrix * sol - rhs))

    push!(data,
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
