import matplotlib.pyplot as plt
import numpy as np

import platform
import os
import sys
from datetime import datetime

import torch

if platform.system() == "Windows":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from analysis.mean_cos_sim import mean_cos_sim
from analysis.correlation import *

def generate_attention_map(aes, analysis_func, acts=[], encoder_or_decoder="decoder", models=None, device='cpu'):
    outputs = []

    for i, ae1 in enumerate(aes):
        row = []
        for j, ae2 in enumerate(aes):
            if analysis_func == 'mcs':
                row.append(mean_cos_sim(ae1, ae2, encoder_or_decoder=encoder_or_decoder)[0])
            elif analysis_func == 'corr':
                row.append(get_correlation_from_acts(acts[i], acts[j], device=device))
            else:
                raise ValueError('analysis_func must be either mcs or corr')
        outputs.append(row)
    return np.array(outputs)


# Create a function to plot the attention maps
def plot_maps(files,  row_labels, col_labels, analysis_func, encoder_or_decoder="decoder", path='./plots/plot', title="NA", n_layers=1, n_maps=1, models=None, device='cpu'):
    text_fontsize = 10
    
    data = get_data(30, 10000)

    # print(len("\n".join(data)))
    data = ["\n".join(data)]
    acts = []
    if analysis_func == 'corr':
        for i, model in enumerate(models):
            acts.append(get_activations(model, files[i], data, device=device))

    fig, axes = plt.subplots(n_layers, n_maps, figsize=(9, 6))
    if n_layers == 1 or n_maps == 1:
        axes = np.array(axes).reshape(n_layers, n_maps)
    #aes = [torch.load(file, map_location=torch.device(device)) for file in files]
    for i in range(n_layers):
        for j in range(n_maps):
            data = generate_attention_map(files, analysis_func, acts = acts, encoder_or_decoder=encoder_or_decoder, models=models, device=device)  # To make the example deterministic
            print(data)
            ax = axes[i, j]
            cax = ax.matshow(data, cmap='YlGn', vmin=0, vmax=1)
            for (x, y), value in np.ndenumerate(data):
                ax.text(y, x, f"{value:.3f}", va='center', ha='center', fontsize=text_fontsize)
            
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=text_fontsize)
            
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=text_fontsize)
            ax.xaxis.set_label_position('bottom')  # Position x-labels at the bottom
            ax.xaxis.tick_bottom()
            
            ax.set_title(title)

    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

# Generate and plot the attention maps
# plot_attention_maps()

# file1 = "./trained_models\EleutherAI_pythia-70m\openwebtext-100k_s0\layer_0\L0_4096_1212-182713.pt"
file1 = "./trained_models/EleutherAI_pythia-70m/openwebtext-100k_s0/layer_0/L0_4096_1212-182713.pt"
file2 = "./trained_models/EleutherAI_pythia-160m/openwebtext-100k_s0/layer_0/L0_6144_1211-104428.pt" #"./trained_models\EleutherAI_pythia-160m\openwebtext-100k_s0\layer_0\L0_6144_1211-104428.pt"
file3 = "./trained_models/EleutherAI_pythia-410m/openwebtext-100k_s0/layer_0/L0_8192_1211-150437.pt" #"./trained_models\EleutherAI_pythia-410m\openwebtext-100k_s0\layer_0\L0_8192_1211-150437.pt"

files = [file1, file2, file3]
row_labels = ["Pythia-70m 1:8", "Pythia-160m 1:8", "Pythia-410m 1:8"]#[file.split('//')[-1][:-15] for file in files]
col_labels = [file.split('//')[-1][:-15] for file in files]

plot_title = "1:8 AEs trained on 3 model sizes (layer 0)"

model_names = ("EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m")
start_time = datetime.now().strftime("%m%d-%H%M%S")

save_path = f"./plots/plot_{start_time}"
plot_maps(files, row_labels, col_labels, 'corr', path=save_path, title = plot_title, n_layers=1, n_maps=1,  models=model_names, device='cuda')
