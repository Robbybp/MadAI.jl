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

for iterate in iterates
    # TODO: Make sure this is right
    # Values must use MadNLP's *reformulated* ordering.
    MadNLP.full(MadNLP.get_x(madnlp)) .= x
    MadNLP.get_y(madnlp) .= y
    MadNLP.get_zl_r(madnlp) .= zl
    MadNLP.get_zu_r(madnlp) .= zu
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
