import platform
import os
import sys
import math
# import einops
# from einops import rearrange, reduce, repeat
import typing as T
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
# from transformer_lens import utils

if platform.system() == "Windows":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dictionary import AutoEncoder

from nnsight import LanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_encoded_acts(model, prompt, ae, layer, device=device):
    """gets activations for a given prompt and layer"""
    result = []
    with model.invoke(prompt):
        result.append(model.gpt_neox.layers[layer].mlp.output.save())
    model_acts = result[0].value.to(device).squeeze()
    return ae.encode(model_acts).squeeze()


def compute_max_average_correlation(small_acts, big_acts, device='cpu'):
    """Compute the max average correlation between small and big activations"""
    epsilon = 1e-5
    # Normalize small_acts
    small_acts_mean = torch.mean(small_acts, dim=0, keepdim=True)
    small_acts_std = torch.std(small_acts, dim=0, keepdim=True)
    small_acts = (small_acts - small_acts_mean) / (small_acts_std + epsilon)
    # Normalize big_acts
    big_acts_mean = torch.mean(big_acts, dim=0, keepdim=True)
    big_acts_std = torch.std(big_acts, dim=0, keepdim=True)
    big_acts = (big_acts - big_acts_mean) / (big_acts_std + epsilon)
    covariance_matrix = torch.mm(small_acts.T, big_acts)
    denominator = torch.sqrt(
        torch.sum(small_acts**2, dim=0, keepdim=True).T
        * torch.sum(big_acts**2, dim=0, keepdim=True)
    )
    pearson_correlation = covariance_matrix / (denominator + epsilon)
    pearson_correlation = torch.where(
        pearson_correlation == 0, torch.tensor(int(-2000)), pearson_correlation
    )
    max_correlations = torch.max(pearson_correlation, dim=1).values
    mask = max_correlations != -2000
    max_correlations = max_correlations[mask]
    avg_max_correlation = torch.mean(max_correlations)
    return avg_max_correlation.item()


def activation_correlations(model_name, path1, path2, layer1=0, layer2=0, device='cpu'):
    """returns avg max correlation between 2 autoencoders"""
    if isinstance(model_name, tuple):
        model1 = LanguageModel(
            model_name[0],  # this can be any Huggingface model
            device_map=device,
        )
        model2 = LanguageModel(
            model_name[1],  # this can be any Huggingface model
            device_map=device,
        )
    else:
        model = LanguageModel(
            model_name,  # this can be any Huggingface model
            device_map=device,
        )
        model1, model2 = model, model
    dataset = load_dataset("NeelNanda/pile-10k", split="train[:30]")
    data = [item["text"][:1000] for item in dataset]
    first_ae = torch.load(path1, map_location=torch.device(device))
    second_ae = torch.load(path2, map_location=torch.device(device))
    avg_over_prompts = []
    for prompt in data:
        # activation shape: (tokens in prompt) x activation_dim
        first_acts = get_encoded_acts(model1, prompt, first_ae, 0, device=device)
        second_acts = get_encoded_acts(model2, prompt, second_ae, 0, device=device)
        avg_max_correlation = compute_max_average_correlation(first_acts, second_acts)
        avg_over_prompts.append(avg_max_correlation)
    return np.mean(avg_over_prompts)


def get_data(num_rows, row_len):
    dataset = load_dataset("NeelNanda/pile-10k", split=f"train[:{num_rows}]")
    data = [item["text"][:row_len] for item in dataset]

    return data


def get_activations(model_name, path, data, device='cpu'):
    """returns avg max correlation between 2 autoencoders"""
    model = LanguageModel(
        model_name,  # this can be any Huggingface model
        device_map=device,
    )
    
    ae = torch.load(path, map_location=torch.device(device))

    acts = []
    for prompt in data:
        # activation shape: (tokens in prompt) x activation_dim
        acts.append(get_encoded_acts(model, prompt, ae, 0, device=device))

    return acts

def get_correlation_from_acts(acts1, acts2, device):
    avg_over_prompts = []
    assert(len(acts1) == len(acts2)), f"expected activations to be the same size but {len(acts1)} != {len(acts2)}"
    for i in range(len(acts1)):
        # activation shape: (tokens in prompt) x activation_dim
        avg_max_correlation = compute_max_average_correlation(acts1[i], acts2[i], device=device)
        avg_over_prompts.append(avg_max_correlation)
    return np.mean(avg_over_prompts)


if __name__ == "__main__":
    model_name = "EleutherAI/pythia-70m"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path1 = './trained_models/EleutherAI_pythia-70m/openwebtext-100k_s0/layer_0/L0_2048_1212-191406_D1.pt'
    path2 = './trained_models/EleutherAI_pythia-70m/openwebtext-100k_s0/layer_0/L0_2048_1212-194010_D2.pt'
    corr = activation_correlations(model_name, path1, path2, device=device)
    print(f'method 1 corr: {corr}')
    
    
    data = get_data(30, 1000)

    print(len("\n".join(data)))
    acts1 = get_activations(model_name, path1, data, device=device)
    acts2 = get_activations(model_name, path2, data, device=device)

    corr = get_correlation_from_acts(acts1, acts2, device=device)

    print(f"method 2 corr: {corr}")