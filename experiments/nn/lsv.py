import torch
import os

INPUT_DIM = 423
OUTPUT_DIM = 186

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "weights")

def create_nn_architecture(nnodes, nlayers, load_weights=False):
    normlayer = torch.nn.Linear(INPUT_DIM, INPUT_DIM)
    network_layers = [normlayer]
    input_dim = INPUT_DIM
    for _ in range(nlayers):
        network_layers.extend((torch.nn.Linear(input_dim, nnodes), torch.nn.Sigmoid()))
        input_dim = nnodes
    network_layers.append(torch.nn.Linear(input_dim, OUTPUT_DIM))
    nn = torch.nn.Sequential(*network_layers)
    if load_weights:
        fname = f"lsv-sigmoid{nnodes}nodes{nlayers}layers.pt"
        weights_path = os.path.join(WEIGHTS_DIR, fname)
        weights = torch.load(weights_path)
        nn.load_state_dict(weights)
    return nn
