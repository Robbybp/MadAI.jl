import PythonCall
torch = PythonCall.pyimport("torch")
module MNIST
include("mnist.jl")
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
    else
        error("Unrecognized model name: $modelname")
    end
end

function get_model(modelname, nodes, layers, kwds...)
    nn = get_nn(modelname, nodes, layers)
    dir = mktempdir()
    nnfile = joinpath(dir, "temp.pt")
    println("Temporarily saving full NN model to $nnfile")
    torch.save(nn, nnfile)
    return get_model(modelname, nnfile; kwds...)
end
