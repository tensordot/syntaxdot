#!/usr/bin/env python3

import argparse
import re
import sys

import h5py
import torch

parser = argparse.ArgumentParser(
    description='Convert a PyTorch SqueezeBert checkpoint to HDF5.')
parser.add_argument(
    'model',
    metavar='MODEL',
    help='The model path')
parser.add_argument('hdf5', metavar='HDF5', help='HDF5 output')

if __name__ == "__main__":
    args = parser.parse_args()

    model = torch.load(args.model)

    with h5py.File(args.hdf5, "w") as hdf5:
        ignore = re.compile("cls|pooler")
        kernel = re.compile("(key|query|value|dense)/weight")
        for var, tensor in model.items():
            # Skip unneeded layers
            if ignore.search(var):
                continue

            var = var.replace("transformer", "squeeze_bert")
            var = var.replace("embeddings.weight", "embeddings")
            var = var.replace("encoder.layers.", "encoder.layer_")
            var = var.replace(".", "/")

            print("Adding %s..." % var, file=sys.stderr)

            # Store the tensor in the HDF5 file.
            hdf5.create_dataset(var, data=tensor)
