import JSON
import NLPModelsJuMP
import MadNLP

include("JuMP/models.jl")

modelname = "mnist"
nodes = 128
layers = 4
iter_no = 1

DATADIR = joinpath(@__DIR__, "data", "iterates")
fname = "$modelname-$(nodes)nodes$(layers)layers.json"
fpath = joinpath(DATADIR, fname)
iterates = open(fpath, "r") do io
    return JSON.parse(io)
end
iterate = iterates[iter_no]
model, formulation = get_model(modelname, nodes, layers)
nlp = NLPModelsJuMP.MathOptNLPModel(model)

# TODO: Options and print level
madnlp = MadNLP.MadNLPSolver(nlp)
# TODO:
# - Initialize solver with primal-dual iterate and barrier parameter
#   (Do I need to worry about bound multipliers? Probably not for the symmetric
#   KKT matrix. But these do show up on the diagonal of the KKT matrix...)
# - Evaluate KKT matrix and RHS
#

MadNLP.initialize!(madnlp)

# TODO: Some method like this should really be part of NLPModelsJuMP
variables, constraints = MadAI.get_var_con_order(model)

# I think I prefer the implementation below
#function _get_vectors(iterate)
#    # Map variables to the value specified
#    x = map(var -> iterate["primal"][JuMP.name(var)], variables)
#    y = map(con -> iterate["dual"][JuMP.name(con)], constraints)
#    zL = map(var -> iterate["Ldual"][JuMP.name(var)], variables)
#    zU = map(var -> iterate["Udual"][JuMP.name(var)], variables)
#    return (; x, y, zL, zU)
#end

# Alternatively:
nlp_varnames = JuMP.name.(variables)
nlp_connames = JuMP.name.(constraints)
primal_order = indexin(iterate_varnames, nlp_varnames)
dual_order = indexin(iterate_connames, nlp_connames)
function _get_vectors(iterate)
    x = iterate["primal"][primal_order]
    y = iterate["dual"][dual_order]
    zL = iterate["Ldual"][primal_order]
    zU = iterate["Udual"][primal_order]
    return (; x, y, zL, zU)
end

for iterate in iterates
    x, y, zL, zU = _get_vectors(iterate)
    μ = iterate["barrier"]

    # TODO: Make sure this is right
    # Values must use MadNLP's *reformulated* ordering.
    MadNLP.full(MadNLP.get_x(madnlp)) .= x
    MadNLP.get_y(madnlp) .= y
    MadNLP.get_zl_r(madnlp) .= zL
    MadNLP.get_zu_r(madnlp) .= zU
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

    K = MadNLP.get_kkt(MadNLP.get_kkt(madnlp))
    rhs = MadNLP.primal_dual(MadNLP.get_p(madnlp))
end
