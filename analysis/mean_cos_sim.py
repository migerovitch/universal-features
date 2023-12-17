import torch

def get_cosine_matrix_updated(weights1, weights2):
    # Normalize each row in weights1 and weights2
    weights1_norm = torch.nn.functional.normalize(weights1, dim=1)
    weights2_norm = torch.nn.functional.normalize(weights2, dim=1)

    # print(weights1_norm.shape, weights2_norm.shape)
    # Compute the cosine similarity matrix
    cos_matrix = torch.mm(weights1_norm, weights2_norm.transpose(0, 1))

    return cos_matrix

def mean_cos_sim(f1, f2, device='cpu', encoder_or_decoder='decoder', second_random=False):
    f1 = torch.load(f1, map_location=torch.device(device))
    f2 = torch.load(f2, map_location=torch.device(device))

    if encoder_or_decoder == 'encoder':
        weights1 = f1.encoder
        weights2 = f2.encoder
    elif encoder_or_decoder == 'decoder':
        weights1 = f1.decoder
        weights2 = f2.decoder
    else:
        raise ValueError("encoder_or_decoder must be 'encoder' or 'decoder'")

    if second_random:
        weights2 = torch.rand_like(weights2)
    
    cos_matrix = get_cosine_matrix_updated(weights1, weights2)
    max_cos_L = torch.max(cos_matrix, dim=1).values
    max_cos_R = torch.max(cos_matrix, dim=0).values

    mean_L = torch.mean(max_cos_L).item()
    mean_R = torch.mean(max_cos_R).item()
    return mean_L, mean_R