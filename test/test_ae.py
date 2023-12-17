import platform
import os
import sys

import typing as T
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import numpy as np

if platform.system() == "Windows":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dictionary import AutoEncoder


from nnsight import LanguageModel
import nnsight

# from buffer import ActivationBuffer
# from training import trainSAE


def load_autoencoder(ae_layer,
                    activation_dim: int = 512,
                    dictionary_size: int = 32768, 
                    layer: int = 0, 
                    device: str ='cpu',
                    version = None,
                    train_filepath = None,
                    checkpoint = None):
    ae = AutoEncoder(activation_dim, dictionary_size)

    version_str = "" if version is None else f"v{version}"
    checkpoint_str = "" if checkpoint is None else f"_c{checkpoint}"

    filepath = f"./data/layer_{layer}/ae_{dictionary_size}{version_str}{checkpoint_str}.pt" if train_filepath is None else train_filepath

    assert os.path.exists(filepath), f"Autoencoder at {filepath} not found"
    if train_filepath is not None:
        ae = torch.load(filepath, map_location=torch.device(device))
    else:
        ae.load_state_dict(torch.load(filepath, map_location=torch.device(device)))
    if ae_layer == 'encoder':
        if train_filepath is None:
            e_1 = ae.encoder.weight
        else:
            e_1 = ae.encoder
    elif ae_layer == 'decoder':
        if train_filepath is None:
            e_1 = ae.decoder.weight.transpose(0, 1)
        else:
            e_1 = ae.decoder
    else:
        raise ValueError("layer must be 'encoder' or 'decoder'")
    
    return e_1


def get_cosine_matrix_updated(weights1, weights2):
    # Normalize each row in weights1 and weights2
    weights1_norm = torch.nn.functional.normalize(weights1, dim=1)
    weights2_norm = torch.nn.functional.normalize(weights2, dim=1)

    # print(weights1_norm.shape, weights2_norm.shape)
    # Compute the cosine similarity matrix
    cos_matrix = torch.mm(weights1_norm, weights2_norm.transpose(0, 1))

    return cos_matrix

from scipy.optimize import linear_sum_assignment

def get_cosine_matrix_hungary(weights1, weights2):
    # Calculate all cosine similarities and store in a 2D array
    print(weights1.shape, weights2.shape)
    cos_sims = np.zeros((weights1.shape[0], weights2.shape[0]))
    for idx, vector in tqdm(enumerate(weights1)):
        expanded_vector = vector.unsqueeze(0)

        cos_sims[idx] = torch.nn.functional.cosine_similarity(expanded_vector, weights2, dim=1).detach().numpy()
    # Convert to a minimization problem
    cos_sims = 1 - cos_sims
    # Use the Hungarian algorithm to solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cos_sims)
    # Retrieve the max cosine similarities and corresponding indices
    max_cosine_similarities = 1 - cos_sims[row_ind, col_ind]
    return max_cosine_similarities

def test_hungary():
    ae_small = load_autoencoder(dictionary_size=8192)
    ae_large = load_autoencoder(dictionary_size=32768)

    max1 = get_cosine_matrix_hungary(ae_small.encoder.weight, ae_large.encoder.weight)
    max2 = get_cosine_matrix_updated(ae_small.encoder.weight, ae_large.encoder.weight)

    print(max1, max2)

    print(np.mean(max1))
    print(torch.mean(max2))




def get_max_cosine_similarity(ae_layer, 
                              max_cols_or_rows = "rows", 
                              ae_dim1 =  8192, 
                              ae_dim2 = 32768,
                              checkpoint = (None, None),
                              version = (None, None), 
                              activation_dim: int = 512, 
                              layer: int = 0, 
                              device: str ='cpu',
                              filepath = (None, None)):
    e_1 = load_autoencoder(ae_layer=ae_layer,
                                activation_dim=activation_dim, 
                                dictionary_size=ae_dim1, 
                                layer=layer, 
                                device=device, 
                                checkpoint=checkpoint[0], 
                                version=version[0],
                                train_filepath=filepath[0])
    e_2 = load_autoencoder(ae_layer=ae_layer,
                                activation_dim=activation_dim, 
                                dictionary_size=ae_dim2, 
                                layer=layer, 
                                device=device, 
                                checkpoint=checkpoint[1], 
                                version=version[1],
                                train_filepath=filepath[1])


    # change e_2 to be random instead of the large autoencoder
    # e_2 = torch.randn(e_2.shape)

    if max_cols_or_rows == "cols":
        e_1, e_2 = e_2, e_1
    elif max_cols_or_rows != "rows":
        raise ValueError("max_cols_or_rows must be 'cols' or 'rows'")
    
    #e_2 = torch.randn(e_2.shape)
    
    cos_matrix = get_cosine_matrix_updated(e_1, e_2)

    # get max of each row
    max_cos = torch.max(cos_matrix, dim=1).values

    print(f"\nlayer: {layer} | {ae_dim1} x {ae_dim2} | {ae_layer} | {max_cols_or_rows}")
    data_mean = torch.mean(max_cos)
    print(f"mean: {data_mean}")
    # print(f"above 0.8: {torch.sum(max_cos > 0.8)}")

    return data_mean
    
#get_max_cosine_similarity(ae_layer = 'encoder', layer=0, ae_dim1 = 8192, ae_dim2 = 8192)


# get_max_cosine_similarity(ae_layer = 'decoder')
# get_max_cosine_similarity(ae_layer = 'decoder', max_cols_or_rows = "cols")



def big_experiment(layers, ae_dim1, ae_dim2, checkpoint= (None, None), version= (None, None), filepath=(None, None)):
    encoder_means_L_to_R = []
    encoder_means_R_to_L = []
    decoder_means_L_to_R = []
    decoder_means_R_to_L = []

    for layer in layers:
        output_1 = encoder_means_L_to_R
        output_2 = encoder_means_R_to_L
        for ae_layer in ['encoder', 'decoder']:
            if ae_layer == 'encoder':
                output_1 = encoder_means_L_to_R
                output_2 = encoder_means_R_to_L
            else:
                output_1 = decoder_means_L_to_R
                output_2 = decoder_means_R_to_L
            
            output_1.append(get_max_cosine_similarity(ae_layer = ae_layer,
                                                        layer=layer,
                                                        ae_dim1=ae_dim1,
                                                        ae_dim2=ae_dim2, 
                                                        version=version,
                                                        checkpoint=checkpoint,
                                                        filepath=filepath
                                                        ).item())
            output_2.append(get_max_cosine_similarity(ae_layer = ae_layer, 
                                                        max_cols_or_rows = "cols", 
                                                        layer=layer,
                                                        ae_dim1=ae_dim1,
                                                        ae_dim2=ae_dim2, 
                                                        version=version,
                                                        checkpoint=checkpoint,
                                                        filepath=filepath).item())
            

    print(f"Encoder MCS Mean L to R: {encoder_means_L_to_R}")
    print(f"Encoder MCS Mean R to L: {encoder_means_R_to_L}")
    print(f"Decoder MCS Mean L to R: {decoder_means_L_to_R}")
    print(f"Decoder MCS Mean R to L: {decoder_means_R_to_L}")
filepath_trained="./data/self_trained/70m-dd.pt"
big_experiment(layers = [5],
               ae_dim1=8192, 
               ae_dim2=8192, 
               filepath=('./data/self_trained/70m_s0_L0.pt', './data/self_trained/70m_s1_L0.pt'))

# print(ae_small.encoder.weight.shape, ae_large.encoder.weight.shape)
