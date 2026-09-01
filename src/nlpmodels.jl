import JuMP
import MathOptInterface as MOI
import NLPModels
import NLPModelsJuMP
import MadNLPHSL

_shape(::MOI.VariableIndex) = JuMP.ScalarShape()
_shape(::MOI.ScalarAffineFunction) = JuMP.ScalarShape()
_shape(::MOI.ScalarQuadraticFunction) = JuMP.ScalarShape()
_shape(::MOI.ScalarNonlinearFunction) = JuMP.ScalarShape()
_shape(::MOI.VectorOfVariables) = JuMP.VectorShape()
_shape(::MOI.VectorAffineFunction) = JuMP.VectorShape()
_shape(::MOI.VectorQuadraticFunction) = JuMP.VectorShape()
_shape(::MOI.VectorNonlinearFunction) = JuMP.VectorShape()

function get_var_con_order(
    model::JuMP.Model
)::Tuple{Vector{JuMP.VariableRef}, Vector{JuMP.ConstraintRef}}
    moimodel = JuMP.backend(model)
    var_indices, con_indices = get_var_con_order(moimodel)
    vars = [JuMP.VariableRef(model, i) for i in var_indices]
    cons = Vector{JuMP.ConstraintRef}()
    for idx in con_indices
        fcn = MOI.get(moimodel, MOI.ConstraintFunction(), idx)
        con = JuMP.ConstraintRef(model, idx, _shape(fcn))
        push!(cons, con)
    end
    return vars, cons
end

function get_con_indices(model::MOI.ModelLike)
    linear = Vector{MOI.ConstraintIndex}()
    quadratic = Vector{MOI.ConstraintIndex}()
    nonlinear = Vector{MOI.ConstraintIndex}()
    oracle = Vector{MOI.ConstraintIndex}()
    contypes = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    for (F, S) in contypes
        if F == MOI.VariableIndex
            continue
        end
        indices = MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
        for idx in indices
            # Why am I branching on fcn and not F here. This loop is a bit more
            # convoluted than it needs to be.
            fcn = MOI.get(model, MOI.ConstraintFunction(), idx)
            if fcn isa MOI.ScalarAffineFunction || fcn isa MOI.VectorAffineFunction
                push!(linear, idx)
            elseif fcn isa MOI.ScalarQuadraticFunction || fcn isa MOI.VectorQuadraticFunction
                push!(quadratic, idx)
            elseif fcn isa MOI.ScalarNonlinearFunction
                push!(nonlinear, idx)
            elseif fcn isa MOI.VectorOfVariables && S <: MOI.VectorNonlinearOracle
                push!(oracle, idx)
            else
                error("Unsupported constraint function $F")
            end
        end
    end
    return (; linear, quadratic, nonlinear, oracle)
end

function get_var_con_order(
    model::MOI.ModelLike
)::Tuple{Vector{MOI.VariableIndex}, Vector{MOI.ConstraintIndex}}
    var_indices = MOI.get(model, MOI.ListOfVariableIndices())
    con_indices = get_con_indices(model)
    con_indices = vcat(con_indices...)
    return var_indices, con_indices
end

function get_kkt_indices(model::JuMP.Model, variables::Vector, constraints::Vector)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    moimodel = JuMP.backend(model)
    for con in constraints
        fcn = MOI.get(moimodel, MOI.ConstraintFunction(), JuMP.index(con))
        if fcn isa MOI.AbstractVectorFunction
            throw(ArgumentError(
                "get_kkt_indices does not support vector constraint $(JuMP.index(con))",
            ))
        end
    end
    varorder, conorder = get_var_con_order(model)
    var_idx_map = Dict(var => i for (i, var) in enumerate(varorder))
    con_idx_map = Dict(con => i for (i, con) in enumerate(conorder))
    vindices = [var_idx_map[v] for v in variables]
    cindices = [con_idx_map[c] for c in constraints]
    # As of MadNLP 0.9.0, the constraint indices are stored on the callback
    # rather than in a separate data structure
    #ind_cons = MadNLP.get_index_constraints(nlp)
    cb = MadNLP.create_callback(MadNLP.SparseCallback, nlp)
    nvar = length(varorder)
    ncon = length(conorder)
    nslack = length(cb.ind_ineq)
    kkt_dim = nvar + ncon + nslack
    kkt_vindices = vindices
    kkt_cindices = cindices .+ (nvar + nslack)
    kkt_indices = vcat(kkt_vindices, kkt_cindices)
    return kkt_indices
end

function update_kkt!(
    kkt::MadNLP.AbstractKKTSystem,
    nlp::NLPModels.AbstractNLPModel;
    x = nothing,
)
    # Need to update:
    # - Hessian
    # - Jacobian
    # - Regularization (set to zero? Or leave as default?)
    # - Σ_x, Σ_s (each for upper and lower bounds)
    # For now, I'd like to do the minimum necessary to give me a nonsingular KKT matrix
    hess_values = MadNLP.get_hessian(kkt)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    #x = NLPModels.get_x0(nlp)
    #x = ones(n)
    if x === nothing
        x = ones(n)
    end
    λ = ones(m)

    NLPModels.hess_coord!(nlp, x, λ, hess_values)

    jac_values = MadNLP.get_jacobian(kkt)
    NLPModels.jac_coord!(nlp, x, jac_values)

    #kkt.reg = 0.0
    #kkt.pr_diag = 0.0
    #kkt.du_diag = 0.0
    return
end

function get_kkt(
    model::JuMP.Model;
    Solver=MadNLPHSL.Ma27Solver,
    opt_linear_solver = MadNLP.default_options(Solver),
)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    # get_index_constraints removed in MadNLP 0.9.0
    #ind_cons = MadNLP.get_index_constraints(nlp)
    cb = MadNLP.create_callback(MadNLP.SparseCallback, nlp)
    # As of 0.9.0, create_kkt_system doesn't require ind_cons
    kkt_system = MadNLP.create_kkt_system(
        MadNLP.SparseKKTSystem,
        cb,
        #ind_cons,
        Solver;
        opt_linear_solver,
    )
    MadNLP.initialize!(kkt_system)
    update_kkt!(kkt_system, nlp)
    MadNLP.build_kkt!(kkt_system)
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    return nlp, kkt_system, kkt_matrix
end
