module MadAI

import SparseArrays

import JuMP
import NLPModels
import NLPModelsJuMP
import HSL
import MadNLP
import MadNLPHSL
import MathOptAI
import MathProgIncidence

include("linalg.jl")
include("blockdiagonal.jl")
include("btsolver.jl")
include("formulation.jl")
include("nlpmodels.jl")
include("kkt-partition.jl")
include("ma48.jl")
include("gpu-schur.jl")

end
