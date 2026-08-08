import JuMP
import HSL_jll
import Ipopt
import PythonCall
torch = PythonCall.pyimport("torch")

include("nn/nn.jl")
include("JuMP/models.jl")

# TODO: Expose these in CLI
modelname = "mnist"
nodes = 128
layers = 4

model, formulation = get_model(modelname, nodes, layers)

ipopt = JuMP.optimizer_with_attributes(
    Ipopt.Optimizer,
    "linear_solver" => "ma57",
    "tol" => 1e-6,
)
JuMP.set_optimizer(model, ipopt)
JuMP.optimize!(model)
