import PythonCall
FILEDIR = @__DIR__
PythonCall.pyimport("sys").path.append(FILEDIR)
torch = PythonCall.pyimport("torch")
pymnist = PythonCall.pyimport("train_mnist")
pyscopf = PythonCall.pyimport("scopf")
pylsv = PythonCall.pyimport("lsv")

function get_nn(modelname, nodes, layers)
    if lowercase(modelname) == "mnist"
        # Note that this function returns an inference-ready network with Softmax
        # and weights loaded.
        return pymnist.create_nn_architecture(nodes, layers, "tanh"; load_weights = true)
    elseif lowercase(modelname) == "scopf"
        return pyscopf.create_nn_architecture(nodes, layers, "tanh"; load_weights = true)
    elseif lowercase(modelname) == "lsv"
        return pylsv.create_nn_architecture(nodes, layers; load_weights = true)
    else
        error("Unexpected model name: $modelname")
    end
end
