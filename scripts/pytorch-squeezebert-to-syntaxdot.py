#!/usr/bin/env python3

import argparse
import re
import sys

import torch

from tensor_module import TensorModule

parser = argparse.ArgumentParser(
    description='Convert a PyTorch SqueezeBERT checkpoint to SyntaxDot tensors.')
parser.add_argument(
    'model',
    metavar='MODEL',
    help='The model path')
parser.add_argument('tensors', metavar='TENSORS', help='SyntaxDot tensors')

if __name__ == "__main__":
    args = parser.parse_args()

    model = torch.load(args.model)

    tensors = {}

    ignore = re.compile("cls|pooler")
    for var, tensor in model.items():
        # Skip unneeded layers
        if ignore.search(var):
            continue

        var = var.replace("transformer.", "")
        var = var.replace("embeddings.weight", "embeddings.embeddings")
        var = var.replace("encoder.layers.", "encoder.layer_")
        var = var.replace("LayerNorm", "layer_norm")
        var = var.replace("layernorm", "layer_norm")

        # Finally, Rust VarStore replaces periods by vertical bars
        # during saving.
        var = var.replace(".", "|")

        print("Adding %s..." % var, file=sys.stderr)

        tensors[var] = tensor

    wrapper = TensorModule(tensors)
    script = torch.jit.script(wrapper)
    script.save(args.tensors)
