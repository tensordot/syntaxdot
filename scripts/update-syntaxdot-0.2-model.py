#!/usr/bin/env python3

import argparse
import os
import re
import sys

import torch

from tensor_module import TensorModule

parser = argparse.ArgumentParser(
    description='Update a SyntaxDot 0.2 BERT/RoBERTa-based model.')
parser.add_argument(
    'model',
    metavar='MODEL',
    help='The model path')
parser.add_argument('converted_model', metavar='CONVERTED_MODEL', help='The converted model')

if __name__ == "__main__":
    args = parser.parse_args()

    model = torch.jit.load(args.model, map_location=torch.device('cpu'))

    tensors = {}

    embeddings_re = re.compile("^encoder\|([^_]+_embeddings)")
    embeddings_layernorm_re = re.compile("^encoder\|layer_norm")
    for var, tensor in model.named_parameters():
        var = embeddings_re.sub(r"embeddings|\1", var)
        var = embeddings_layernorm_re.sub(r"embeddings|layer_norm", var)
        var = var.replace("encoder|token_type_embeddings", "embeddings|token_type_embeddings")

        tensors[var] = tensor


    wrapper = TensorModule(tensors)
    script = torch.jit.script(wrapper)
    script.save(args.converted_model)
