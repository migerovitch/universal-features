import matplotlib.pyplot as plt
import numpy as np

import platform
import os
import sys

import torch

if platform.system() == "Windows":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from analysis.mean_cos_sim import mean_cos_sim

def plot_bars(file_groups, encoder_or_decoder="decoder"):
    groups = len(file_groups)
    bars_per_group = len(file_groups[0])
    data = np.zeros((groups, bars_per_group))

    for i, group in enumerate(file_groups):
        for j, pair in enumerate(group):
            data[i][j] = mean_cos_sim(pair[0], pair[1], encoder_or_decoder=encoder_or_decoder, second_random=True)[0]

    # Setting up the figure and the axes
    fig, ax = plt.subplots()

    # Setting the positions for each group on the X-axis
    indices = np.arange(groups)

    # Width of a bar
    bar_width = 0.2

    # Plotting the bars
    for i in range(bars_per_group):
        ax.bar(indices + i * bar_width, data[:, i], bar_width, label=f'Bar {i+1}')

    # Adding labels and title
    ax.set_xlabel('Group')
    ax.set_ylabel('Values')
    ax.set_title('6 Groups of Three Bars')
    ax.set_xticks(indices + bar_width)
    ax.set_xticklabels([f'Group {i+1}' for i in range(groups)])
    ax.legend()

    # Displaying the plot
    plt.show()

L0 = [('./trained_models\EleutherAI_pythia-70m-deduped\openwebtext-100k_s0\layer_0\L0_2048_1212-234643.pt',
       './trained_models\EleutherAI_pythia-70m\openwebtext-100k_s0\layer_0\L0_4096_1212-182713.pt'),
       ('./trained_models\EleutherAI_pythia-160m-deduped\openwebtext-100k_s0\layer_0\L0_3072_1211-100253.pt',
        './trained_models\EleutherAI_pythia-160m\openwebtext-100k_s0\layer_0\L0_3072_1211-092238.pt'),
        ('./trained_models\EleutherAI_pythia-410m-deduped\openwebtext-100k_s0\layer_0\L0_4096_1211-134736.pt',
         './trained_models\EleutherAI_pythia-410m\openwebtext-100k_s0\layer_0\L0_4096_1211-122353.pt')]

L1 = [('./trained_models\EleutherAI_pythia-70m-deduped\openwebtext-100k_s0\layer_1\L1_2048_1212-234643.pt', 
        './trained_models\EleutherAI_pythia-70m\openwebtext-100k_s0\layer_1\L1_4096_1212-182713.pt'),
        ('./trained_models\EleutherAI_pythia-160m-deduped\openwebtext-100k_s0\layer_1\L1_3072_1211-100253.pt',
         './trained_models\EleutherAI_pythia-160m\openwebtext-100k_s0\layer_1\L1_3072_1211-092238.pt'),
         ('./trained_models\EleutherAI_pythia-410m-deduped\openwebtext-100k_s0\layer_1\L1_4096_1211-134736.pt',
          './trained_models\EleutherAI_pythia-410m\openwebtext-100k_s0\layer_1\L1_4096_1211-122353.pt')
        ]

L2 = [('./trained_models\EleutherAI_pythia-70m-deduped\openwebtext-100k_s0\layer_2\L2_2048_1212-234643.pt',
         './trained_models\EleutherAI_pythia-70m\openwebtext-100k_s0\layer_2\L2_4096_1212-182713.pt'),
         ('./trained_models\EleutherAI_pythia-160m-deduped\openwebtext-100k_s0\layer_2\L2_3072_1211-100253.pt',
          './trained_models\EleutherAI_pythia-160m\openwebtext-100k_s0\layer_2\L2_3072_1211-092238.pt'),
          ('./trained_models\EleutherAI_pythia-410m-deduped\openwebtext-100k_s0\layer_2\L2_4096_1211-134736.pt',
           './trained_models\EleutherAI_pythia-410m\openwebtext-100k_s0\layer_2\L2_4096_1211-122353.pt')]

L3 = [('./trained_models\EleutherAI_pythia-70m-deduped\openwebtext-100k_s0\layer_3\L3_2048_1212-234643.pt',
       './trained_models\EleutherAI_pythia-70m\openwebtext-100k_s0\layer_3\L3_4096_1212-182713.pt'),
       ('./trained_models\EleutherAI_pythia-160m-deduped\openwebtext-100k_s0\layer_3\L3_3072_1211-100253.pt',
        './trained_models\EleutherAI_pythia-160m\openwebtext-100k_s0\layer_3\L3_3072_1211-092238.pt'),
        ('./trained_models\EleutherAI_pythia-410m-deduped\openwebtext-100k_s0\layer_3\L3_4096_1211-134736.pt',
         './trained_models\EleutherAI_pythia-410m\openwebtext-100k_s0\layer_3\L3_4096_1211-122353.pt')]

L4 = [('./trained_models\EleutherAI_pythia-70m-deduped\openwebtext-100k_s0\layer_4\L4_2048_1212-234643.pt',
       './trained_models\EleutherAI_pythia-70m\openwebtext-100k_s0\layer_4\L4_4096_1212-182713.pt'),
       ('./trained_models\EleutherAI_pythia-160m-deduped\openwebtext-100k_s0\layer_4\L4_3072_1211-100253.pt',
        './trained_models\EleutherAI_pythia-160m\openwebtext-100k_s0\layer_4\L4_3072_1211-092238.pt'),
        ('./trained_models\EleutherAI_pythia-410m-deduped\openwebtext-100k_s0\layer_4\L4_4096_1211-134736.pt',
         './trained_models\EleutherAI_pythia-410m\openwebtext-100k_s0\layer_4\L4_4096_1211-122353.pt')]

L5 = [('./trained_models\EleutherAI_pythia-70m-deduped\openwebtext-100k_s0\layer_5\L5_2048_1212-234643.pt',
       './trained_models\EleutherAI_pythia-70m\openwebtext-100k_s0\layer_5\L5_4096_1212-182713.pt'),
       ('./trained_models\EleutherAI_pythia-160m-deduped\openwebtext-100k_s0\layer_5\L5_3072_1211-100253.pt',
        './trained_models\EleutherAI_pythia-160m\openwebtext-100k_s0\layer_5\L5_3072_1211-092238.pt'),
        ('./trained_models\EleutherAI_pythia-410m-deduped\openwebtext-100k_s0\layer_5\L5_4096_1211-134736.pt',
         './trained_models\EleutherAI_pythia-410m\openwebtext-100k_s0\layer_5\L5_4096_1211-122353.pt')]

plot_bars([L0, L1, L2, L3, L4, L5])