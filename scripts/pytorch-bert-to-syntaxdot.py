#!/usr/bin/env python3

import argparse
import os
import re
import sys

import torch

from tensor_module import TensorModule

parser = argparse.ArgumentParser(
    description='Convert PyTorch BERT checkpoint to SyntaxDot tensors.')
parser.add_argument(
    'model',
    metavar='MODEL',
    help='The model path')
parser.add_argument('tensors', metavar='TENSORS', help='SyntaxDot tensors')
parser.add_argument('--albert', action='store_true', default=False, help="Convert an ALBERT model")

if __name__ == "__main__":
    args = parser.parse_args()

    model = torch.load(args.model)

    tensors = {}

    ignore = re.compile("adam_v|adam_m|global_step|cls|pooler|position_ids")
    self_attention = re.compile("attention\.(key|query|value)")
    for var, tensor in model.items():
        # Skip unneeded layers
        if ignore.search(var):
            continue

        # Remove prefix
        if args.albert:
            var = var.replace("albert.", "")
            var = var.replace("albert_", "")
        else:
            var = var.replace("bert.", "")

        # Rewrite some variable names
        var = var.replace("embeddings.weight", "embeddings.embeddings")
        var = var.replace("kernel", "weight")
        var = var.replace("gamma", "weight")
        var = var.replace("beta", "bias")
        var = var.replace("LayerNorm", "layer_norm")
        var = var.replace("layer.", "layer_")

        if args.albert:
            var = var.replace("layer_groups.", "group_")
            var = var.replace("layers.", "inner_group_")
            var = var.replace("embedding_hidden_mapping_in", "embedding_projection")
            var = self_attention.sub(r"attention.self.\1", var)
            var = var.replace("attention.dense", "attention.output.dense")
            var = var.replace("attention.layer_norm", "attention.output.layer_norm")
            var = var.replace("ffn.", "intermediate.dense.")
            var = var.replace("ffn_output", "output.dense")
            var = var.replace("full_layer_layer_norm", "output.layer_norm")

        # Finally, Rust VarStore replaces periods by vertical bars
        # during saving.
        var = var.replace(".", "|")

        print("Adding %s..." % var, file=sys.stderr)

        tensors[var] = tensor

    wrapper = TensorModule(tensors)
    script = torch.jit.script(wrapper)
    script.save(args.tensors)
