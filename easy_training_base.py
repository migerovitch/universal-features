#!/usr/bin/env python
# coding: utf-8

import torch 
import argparse
from utils import dotdict
from activation_dataset import setup_token_data
import wandb
import json
from datetime import datetime
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import os
from baukit import Trace, TraceDict
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, GPTJForSequenceClassification

from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE

def init_cfg():
    cfg = dotdict()
    # models: "EleutherAI/pythia-6.9b", "lomahony/eleuther-pythia6.9b-hh-sft", "usvsnsp/pythia-6.9b-ppo", "Dahoas/gptj-rm-static", "reciprocate/dahoas-gptj-rm-static"
    # cfg.model_name="lomahony/eleuther-pythia6.9b-hh-sft"
    # "EleutherAI/pythia-70m", "lomahony/pythia-70m-helpful-sft", "lomahony/eleuther-pythia70m-hh-sft"
    cfg.model_name="EleutherAI/pythia-70m-deduped"
    cfg.layers=[0, 1] # Change this to run multiple layers
    cfg.setting="residual"
    # cfg.tensor_name="gpt_neox.layers.{layer}" or "transformer.h.{layer}"
    cfg.tensor_name="gpt_neox.layers.{layer}.mlp"
    original_l1_alpha = 8e-3 # originally 8e-4
    cfg.l1_alpha=original_l1_alpha
    cfg.l1_alphas=[8e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1e-3, 2e-3, 4e-3, 8e-3]
    cfg.sparsity=None
    cfg.num_epochs=2
    cfg.model_batch_size=8
    cfg.lr=1e-3 # ORIGINAL: 1e-3
    cfg.kl=False
    cfg.reconstruction=False
    #cfg.dataset_name="NeelNanda/pile-10k"
    cfg.dataset_name="Elriggs/openwebtext-100k"
    cfg.device="cuda:0"
    cfg.ratio = 8
    cfg.seed = 0
    # cfg.device="cpu"

    return cfg


# # Main Code

# In[35]:


def setup_execute_training(model_name,
                          dataset_name,
                          ratio,
                          layers,
                          seed,
                          wandb_log,
                          split,
                          epoches,
                          wandb_project_name):
    cfg = init_cfg()
    cfg.num_epochs = epoches
    cfg.model_name = model_name
    cfg.dataset_name = dataset_name
    cfg.ratio = ratio
    cfg.layers = layers
    cfg.seed = seed
    cfg.wandb_log = wandb_log

    model, tokenizer = load_model(cfg)
    get_activation_size(cfg, model, tokenizer)

    # naming
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb_project_name = wandb_project_name
    wandb_run_name = f"{cfg.model_name}_{cfg.dataset_name}_dim{cfg.ratio*cfg.activation_size}_{start_time[4:]}"
    model_name_path = cfg.model_name.replace("/", "_")
    dataset_name_path = cfg.dataset_name.split("/")[-1]

    storage_path = f"{model_name_path}/{dataset_name_path}"

    filename = f"{cfg.ratio*cfg.activation_size}_{start_time[4:]}"
    token_loader = init_dataloader(cfg, model, tokenizer, split)
    autoencoders, optimizers = init_autoencoder(cfg)
    
    if wandb_log:
        setup_wandb(cfg, wandb_run_name, wandb_project_name)
    
    training_run(cfg, model, optimizers, autoencoders, token_loader)

    for layer in range(len(cfg.layers)):
        model_save(cfg, autoencoders[layer], storage_path, filename, cfg.layers[layer])



# Load in the model
def load_model(cfg):
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model = model.to(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return model, tokenizer

# Download the dataset
# TODO iteratively grab dataset?
def init_dataloader(cfg, model, tokenizer, split="train"):
    cfg.max_length = 256
    token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed, split=split)
    num_tokens = cfg.max_length*cfg.model_batch_size*len(token_loader)
    print(f"Number of tokens: {num_tokens}")
    cfg.total_tokens = num_tokens
    
    return token_loader

def get_activation_size(cfg, model, tokenizer):
    text = "1"
    tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(cfg.device)
    # Your activation name will be different. In the next cells, we will show you how to find it.
    with torch.no_grad():
        with Trace(model, tensor_names[0]) as ret:
            _ = model(tokens)
            representation = ret.output
            # check if instance tuple
            if(isinstance(representation, tuple)):
                representation = representation[0]
            activation_size = representation.shape[-1]
    print(f"Activation size: {activation_size}")
    cfg.activation_size = activation_size
    return activation_size

def init_autoencoder(cfg):
    autoencoders = []
    optimizers = []
    for layer in range(len(cfg.layers)):
        params = dict()
        n_dict_components = cfg.activation_size*cfg.ratio # Sparse Autoencoder Size
        params["encoder"] = torch.empty((n_dict_components, cfg.activation_size), device=cfg.device)
        nn.init.xavier_uniform_(params["encoder"])
    
        params["decoder"] = torch.empty((n_dict_components, cfg.activation_size), device=cfg.device)
        nn.init.xavier_uniform_(params["decoder"])
    
        params["encoder_bias"] = torch.empty((n_dict_components,), device=cfg.device)
        nn.init.zeros_(params["encoder_bias"])
    
        params["shift_bias"] = torch.empty((cfg.activation_size,), device=cfg.device)
        nn.init.zeros_(params["shift_bias"])
    
        autoencoder = AnthropicSAE(  # TiedSAE, UntiedSAE, AnthropicSAE
            # n_feats = n_dict_components, 
            # activation_size=cfg.activation_size,
            encoder=params["encoder"],
            encoder_bias=params["encoder_bias"],
            decoder=params["decoder"],
            shift_bias=params["shift_bias"],
        )
        autoencoder.to_device(cfg.device)
        autoencoder.set_grad()
    
        optimizer = torch.optim.Adam(
            [
                autoencoder.encoder, 
                autoencoder.encoder_bias,
                autoencoder.decoder,
                autoencoder.shift_bias,
            ], lr=cfg.lr)
        autoencoders.append(autoencoder)
        optimizers.append(optimizer)
    return autoencoders, optimizers

def setup_wandb(cfg, wandb_run_name, wandb_project_name):
    secrets = json.load(open("secrets.json"))
    wandb.login(key=secrets["wandb_key"])
    wandb.init(project=wandb_project_name, config=dict(cfg), name=wandb_run_name)
    return wandb_run_name

def training_run(cfg, model, optimizers, autoencoders, token_loader):

    time_since_activation = torch.zeros(autoencoders[0].encoder.shape[0])
    total_activations = torch.zeros(autoencoders[0].encoder.shape[0])
    tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]
    max_num_tokens = cfg.total_tokens # 100_000_000
    save_every = 30_000
    num_saved_so_far = 0

    # Freeze model parameters 
    model.eval()
    model.requires_grad_(False)
    model.to(cfg.device)
    
    last_encoder = autoencoders[0].encoder.clone().detach()
    assert len(cfg.layers) == len(tensor_names), "layers and tensor_names have different lengths"
    for epoch in range(cfg.num_epochs):
        for i, batch in enumerate(tqdm(token_loader)): #,total=int(max_num_tokens/(cfg.max_length*cfg.model_batch_size)))):
            tokens = batch["input_ids"].to(cfg.device)
            # print(f"tokens shape: {tokens.shape}")
            
            with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model
                
                #print(tensor_names)
                representations = []
                with TraceDict(model, tensor_names) as ret:
                    _ = model(tokens)
                    for tensor_name in tensor_names:
                        representations.append(ret[tensor_name].output)
                    assert not isinstance(representations[0], tuple), "representations is type tuple"
                    # print(len(representations), representations[0].shape)
                    # if(isinstance(representation, tuple)):
                    #     representation = representation[0]
            #print(f"representation is: {representation}")
            #print(f"representation shape is: {representation.shape}")
            
        
            # activation_saver.save_batch(layer_activations.clone().cpu().detach())
            for layer in range(len(cfg.layers)):
                representation = representations[layer]
                layer_activations = rearrange(representation, "b seq d_model -> (b seq) d_model")
                autoencoder = autoencoders[layer]
                optimizer = optimizers[layer]
                
                c = autoencoder.encode(layer_activations)
                x_hat = autoencoder.decode(c)
                
                reconstruction_loss = (x_hat - layer_activations).pow(2).mean()
                l1_loss = torch.norm(c, 1, dim=-1).mean()
                total_loss = reconstruction_loss + cfg.l1_alpha*l1_loss
            
                time_since_activation += 1
                time_since_activation = time_since_activation * (c.sum(dim=0).cpu()==0)
                # total_activations += c.sum(dim=0).cpu()
                if ((i) % 100 == 0): # Check here so first check is model w/o change
                    # self_similarity = torch.cosine_similarity(c, last_encoder, dim=-1).mean().cpu().item()
                    # Above is wrong, should be similarity between encoder and last encoder
                    self_similarity = torch.cosine_similarity(autoencoder.encoder, last_encoder, dim=-1).mean().cpu().item()
                    last_encoder = autoencoder.encoder.clone().detach()
            
                    num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
                    with torch.no_grad():
                        sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
                        # Count number of dead_features are zero
                        num_dead_features = (time_since_activation >= min(i, 200)).sum().item()
                    print(f"Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Total Loss: {total_loss:.4f} | Reconstruction Loss: {reconstruction_loss:.4f} | L1 Loss: {cfg.l1_alpha*l1_loss:.4f} | l1_alpha: {cfg.l1_alpha:.4e} | Tokens: {num_tokens_so_far} | Self Similarity: {self_similarity:.4f}")
                    
                    if cfg.wandb_log:
                        wandb.log({
                            'Sparsity': sparsity,
                            'Dead Features': num_dead_features,
                            'Total Loss': total_loss.item(),
                            'Reconstruction Loss': reconstruction_loss.item(),
                            'L1 Loss': (cfg.l1_alpha*l1_loss).item(),
                            'l1_alpha': cfg.l1_alpha,
                            'Tokens': num_tokens_so_far,
                            'Self Similarity': self_similarity
                        })
                    
                    dead_features = torch.zeros(autoencoder.encoder.shape[0])
                    
                    # if(num_tokens_so_far > max_num_tokens):
                    #     print(f"Reached max number of tokens: {max_num_tokens}")
                    #     break
                    
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
    wandb.finish()

def model_save(cfg, autoencoder, storage_path, filename, layer):
    model_save_name = cfg.model_name.split("/")[-1]

    # start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # save_name = f"{model_save_name}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}_{start_time}"  # trim year
    storage_path_name = "trained_models/" + storage_path + f"/layer_{layer}"
    # Make directory traiend_models if it doesn't exist
    if not os.path.exists(storage_path_name):
        os.makedirs(storage_path_name)
    # Save model
    filename = f"L{layer}_{filename}"
    
    torch.save(autoencoder, f"{storage_path_name}/{filename}.pt")
    print(f"Saved file at: {storage_path_name}/{filename}.pt")

if __name__ == "__main__":
    model_name = "EleutherAI/pythia-70m"
    dataset_name = "Elriggs/openwebtext-100k" # "Elriggs/openwebtext-100k"
    ratio = 1
    layers = [0]
    wandb_log = True
    seed = 0
    split = "train[:1000]"
    epoches = 1
    wandb_project_name = "test_1"

    setup_execute_training(model_name,
                        dataset_name,
                        ratio,
                        layers,
                        seed,
                        wandb_log=wandb_log,
                        split=split,
                        epoches=epoches,
                        wandb_project_name=wandb_project_name)
