#!/usr/bin/env python3

import argparse
import os
import re
import sys

import h5py
import torch

parser = argparse.ArgumentParser(
    description='Convert a PyTorch RoBERTa checkpoint to HDF5.')
parser.add_argument(
    'model',
    metavar='MODEL',
    help='The model path')
parser.add_argument('hdf5', metavar='HDF5', help='HDF5 output')

if __name__ == "__main__":
    args = parser.parse_args()

    model = torch.load(args.model)

    with h5py.File(args.hdf5, "w") as hdf5:
        ignore = re.compile("adam_v|adam_m|global_step|lm_head|pooler")
        kernel = re.compile("(key|query|value|dense)/weight")
        for var, tensor in model.items():
            # Skip unneeded layers
            if ignore.search(var):
                continue

            var = var.replace("roberta", "bert")
            var = var.replace("embeddings.weight", "embeddings")
            var = var.replace("encoder.layer.", "encoder.layer_")
            var = var.replace(".", "/")

            # Attention weight matrices are transposed, compared to BERT.
            if kernel.search(var):
                tensor = tensor.t()

            print("Adding %s..." % var, file=sys.stderr)

            # Store the tensor in the HDF5 file.
            hdf5.create_dataset(var, data=tensor)
