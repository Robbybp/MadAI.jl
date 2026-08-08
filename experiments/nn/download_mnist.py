from torchvision import datasets
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", default=None, help="Directory to store data")
    parser.add_argument("--absolute", action="store_true", help="Path is absolute, instead of relative to file dir")
    # TODO: Implement
    args = parser.parse_args()
    datasets.MNIST(root=os.getcwd(), download=True)
