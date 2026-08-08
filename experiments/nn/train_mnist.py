import argparse
import torch
import torchvision
from torchvision.transforms.v2 import PILToTensor, ToTensor
import os
import time

torch.manual_seed(0)
FILEDIR = os.path.dirname(__file__)
WEIGHTSDIR = os.path.join(FILEDIR, os.pardir, "data", "weights")

ACTIVATION_LOOKUP = {
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "relu": torch.nn.ReLU,
    "softplus": torch.nn.Softplus,
    "softmax": torch.nn.Softmax,
}


def _get_fname(nodes, layers, activation):
    return f"mnist-{activation}{nodes}nodes{layers}layers.pt"


def create_nn_architecture(nodes, layers, activation, softmax=True, load_weights=True):
    """Create an MLP with the requested hidden-layer configuration. By default,
    the network is set up for inference, with the Softmax layer appended and
    weights loaded from the default location.
    """
    # TODO: Move hardcoded dimensions
    input_dim = 28 * 28
    output_dim = 10
    activation_function = ACTIVATION_LOOKUP[activation]
    network_layers = [torch.nn.Linear(input_dim, nodes)]
    for _ in range(layers):
        network_layers.append(activation_function())
        network_layers.append(torch.nn.Linear(nodes, nodes))
    network_layers.append(activation_function())
    network_layers.append(torch.nn.Linear(nodes, output_dim))
    if softmax:
        network_layers.append(torch.nn.Softmax(dim=-1))
    nn = torch.nn.Sequential(*network_layers)
    if load_weights:
        weightfile = os.path.join(WEIGHTSDIR, _get_fname(nodes, layers, activation))
        weights = torch.load(weightfile)
        nn.load_state_dict(weights)
    return nn


def predict(nn, x):
    output = nn(x)
    output_dim = len(output)
    return max(range(output_dim), key=lambda i: output[i])


def evaluate_accuracy(nn, dataset, *, device="cpu"):
    nn.to(device)
    nsamples = len(dataset)
    input_dim = torch.prod(torch.tensor(dataset[0][0].shape))
    x = dataset[0][0].reshape(input_dim).to(device)
    output_dim = len(nn(x))
    inputs = [image.reshape(input_dim) for image, _ in dataset]
    inputs = torch.stack(inputs)
    inputs = inputs.to(device)
    outputs = nn(inputs)
    predictions = torch.argmax(outputs, dim=1)
    labels = torch.tensor([label for _, label in dataset])
    labels = labels.to(device)
    ncorrect = torch.sum(labels == predictions)
    return float(ncorrect / nsamples)


def main(args):
    #transform = PILToTensor() # <- This doesn't scale the tensor
    transform = ToTensor()     # <- This scales the tensor
    train_dataset = torchvision.datasets.MNIST(
        # TODO: make dir CLI-configurable?
        root=FILEDIR,
        train=True,
        transform=transform,
    )

    # "DataLoader combines a dataset and a sampler, and provides an iterable over
    # the given dataset"
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
    )

    _, image_height, image_width = train_dataset.data.shape
    input_dim = image_height * image_width
    # The outputs are raw logits; softmax is omitted for cross-entropy training.
    nn = create_nn_architecture(
        args.nodes,
        args.layers,
        args.activation,
        softmax=False,
        load_weights=False,
    )
    output_dim = 10

    # Example prediction
    if False:
        image, label = train_dataset[100]
        y = nn(image.reshape(input_dim))
        pred = max(range(output_dim), key=lambda i: y[i])
        print("Example prediction on image 100:")
        print(f"output = {y}")
        print(f"Prediction: {pred}")

    # Send model and data to device
    nn.to(args.device)

    compute_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn.parameters(), lr=args.learning_rate)
    nn.train() # Set model in training mode. Not sure why...
    t_train_start = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            inshape = inputs.shape
            # inshape[0] is the batch size, potentially truncated because we're at the
            # end of the data
            inputs = inputs.reshape(inshape[0], input_dim)
            outputs = nn(inputs)
            #labels.to(float)
            #expected = torch.nn.functional.one_hot(labels)
            # I guess CrossEntropyLoss accepts labels directly?
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        ave_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} / {args.epochs}: Loss = {ave_loss:1.2e}")
    t_train = time.time() - t_train_start
    print(f"Time spent training: {t_train:1.2f}")

    # Note that this method of appending Softmax has to match how we append
    # softmax in construct_nn_architecture (whether the original NN is flattened
    # or nested.
    likelihood_predictor = torch.nn.Sequential(*nn.children(), torch.nn.Softmax(dim=-1))
    acc = evaluate_accuracy(nn, train_dataset, device=args.device)
    ntrain = len(train_dataset)
    print(f"Accuracy on training set of {ntrain} samples: {acc}")

    # Evaluate on test data
    test_dataset = torchvision.datasets.MNIST(
        root=FILEDIR,
        transform=transform,
        train=False,
    )
    nn.eval()
    acc = evaluate_accuracy(nn, test_dataset, device=args.device)
    ntest = len(test_dataset)
    print(f"Accuracy on test set of {ntest} samples: {acc}")

    # Send model back to CPU
    nn.to("cpu")

    fname = _get_fname(args.nodes, args.layers, args.activation)
    fpath = os.path.join(WEIGHTSDIR, fname)
    weights = likelihood_predictor.state_dict()
    if args.dry_run:
        print(f"--dry-run set. Not saving. Would have saved weights to {fpath}")
    else:
        print(f"Saving weights to {fpath}")
        torch.save(weights, fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default=128, type=int, help="Nodes per layer. (default=128)")
    parser.add_argument("--layers", default=4, type=int, help="Number of layers. (default=4)")
    parser.add_argument("--activation", default="relu", help=f"Activation function. (default=relu. options={list(ACTIVATION_LOOKUP)})")
    parser.add_argument("--dry-run", action="store_true", help="Don't save trained network")
    parser.add_argument("--device", default="cpu", help="default='cpu'")
    parser.add_argument("--epochs", type=int, default=10, help="default=10")
    parser.add_argument("--batchsize", type=int, default=64, help="default=64")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="default=1e-3")
    args = parser.parse_args()
    main(args)
