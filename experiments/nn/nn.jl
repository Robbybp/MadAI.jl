import PythonCall
FILEDIR = @__DIR__
PythonCall.pyimport("sys").path.append(FILEDIR)
pymnist = PythonCall.pyimport("train_mnist")

function get_nn(modelname, nodes, layers)
    if lowercase(modelname) == "mnist"
        # Note that this function returns an inference-ready network with Softmax
        # and weights loaded.
        return pymnist.create_nn_architecture(nodes, layers, "tanh")
    else
        error("Unexpected model name: $modelname")
    end
end
