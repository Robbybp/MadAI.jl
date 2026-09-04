import PythonCall
torch = PythonCall.pyimport("torch")

module MNIST
include("mnist.jl")
end

module SCOPF
include("scopf/scopf.jl")
end

# This is only necessary for the method where we construct straight from
# name, nodes, and layers.
include("../nn/nn.jl")

function get_model(modelname, nnfile; kwds...)
    if lowercase(modelname) == "mnist"
        image_index = 7; adversarial_label = 4; threshold = 0.7
        m, y, formulation = MNIST.get_adversarial_model(
            nnfile, image_index, adversarial_label, threshold;
            kwds...
        )
        return m, formulation
    elseif lowercase(modelname) == "scopf"
        m, formulation = SCOPF.get_scopf_model(nnfile; kwds...)
        pivot_vars, _ = MadAI.get_vars_cons(formulation)
        for var in pivot_vars
            # Pivot vars should not be fixed (they must exist in the KKT system),
            # so, if they are, we relax the bounds.
            if JuMP.has_lower_bound(var) && JuMP.has_upper_bound(var) && JuMP.lower_bound(var) == JuMP.upper_bound(var)
                JuMP.set_lower_bound(var, JuMP.lower_bound(var) - 1e-6)
                JuMP.set_upper_bound(var, JuMP.upper_bound(var) + 1e-6)
            end
        end
        return m, formulation
    else
        error("Unrecognized model name: $modelname")
    end
end

function get_model(modelname, nodes, layers; kwds...)
    nn = get_nn(modelname, nodes, layers)
    dir = mktempdir()
    nnfile = joinpath(dir, "temp-$modelname-nn.pt")
    println("Temporarily saving full NN model to $nnfile")
    torch.save(nn, nnfile)
    return get_model(modelname, nnfile; kwds...)
end
