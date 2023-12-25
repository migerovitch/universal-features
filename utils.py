import json
import os
import subprocess
import sys
from typing import Optional
from pathlib import Path


class dotdict(dict):
    """Dictionary that can be accessed with dot notation."""

    def __init__(self, d: Optional[dict] = None):
        if d is None:
            d = {}
        super().__init__(d)

    def __dict__(self):
        return self

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Attribute {name} not found")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

def make_tensor_name(cfg):
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        tensor_name = f"blocks.{cfg.layer}.mlp.hook_post"
        if cfg.model_name == "gpt2":
            cfg.mlp_width = 3072
        elif cfg.model_name == "EleutherAI/pythia-70m-deduped":
            cfg.mlp_width = 2048
    elif cfg.model_name == "nanoGPT":
        tensor_name = f"transformer.h.{cfg.layer}.mlp.c_fc"
        cfg.mlp_width = 128
    else:
        raise NotImplementedError(f"Model {cfg.model_name} not supported")

    return tensor_name