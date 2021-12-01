import networkx as nx 
import torch
from torch import nn
import random

def graph_to_edge_list(G, directed = False, self_loops=False):
    """Takes a networkx Graph and returns a list of edges. 
    Edges are tuples 
    
    If directed = True: (1,2) AND (2,1) will appear in list. 
    If selfloops=True, more calculations have to be made
    """
    edge_list = [edge for edge in nx.edges(G)]
    if directed: 
        if self_loops: 
            reverse_edge_list = [reversed(edge) for edge in edge_list if edge[0]!= edge[1]]
        else: 
            reverse_edge_list = [edge[::-1] for edge in edge_list]
        edge_list = edge_list + reverse_edge_list
    return edge_list


def edge_list_to_tensor(edge_list):
    """
    Takes a list of edges (edges are tuples) and converts it to a tensor. 
    
    Returns: 
        - edge_index: tensor of shape (2, number_of_edges)
    """
    edge_index = torch.tensor(edge_list)
    edge_index = torch.transpose(edge_index, 0,1)
    edge_index_reversed = edge_index[[1,0],:]
    return torch.cat([edge_index, edge_index_reversed], dim = 1)


def sample_negative_edges(G, num_neg_samples):
    """Returns list of #num_neg_samples of edges that are not in the graph.
    Edges are represented as tuples. 
    """
    neg_edge_list = random.sample(list(nx.non_edges(G)), num_neg_samples)
    return neg_edge_list


def create_node_emb(num_node, embedding_dim=16):
    """
    Returns nn.Embedding object with #num_node embeddings. 
    Each embedding has dimension of #embedding_dim. 
    Embedding weights are initialized randomly.
    """
    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
    emb.weight.data = torch.rand(emb.weight.data.shape)
    return emb


def transductive_edge_split(edge_list, 
                            split_dict,
                            seed=25
                           ): 
    """
    Returns dictionary with edges split in specified categories 
    (specified in split_dict). Dictionary has same keys as split_dict.
    
    Params: 
        - edge_list: list of all edges that are to be split 
                (should be a list of tuples)
        - split_dict: dictionary with splitting details
                should have format {category: #percentage_of_values_in_category}. 
                'test' category shouldn't be included, will be generated automatically. 
        - seed: integer, random seed for reproducability
    
    """
    random.seed(seed)
    
    num_edges = len(edge_list)
    idx = list(range(num_edges))
    random.shuffle(idx)
    
    # save cutoff values for each key category (e.g. train cutoff 123 and validation cutoff at 156)
    cutoff_dict = dict()
    perc_accum = 0
    for key, perc in split_dict.items():
        perc_accum += perc
        cutoff_dict[key] = int(num_edges * perc_accum)
    
    edges = dict()
    last_cutoff = 0
    for key, cutoff in cutoff_dict.items(): 
        edges[f"{key}"] = [edge_list[i] for i in idx[last_cutoff:cutoff]]
        last_cutoff = cutoff 
    if last_cutoff != num_edges: 
        edges[f"test"] = [edge_list[i] for i in idx[last_cutoff:]]
    return edges


def accuracy(pred, label, threshold=0.5):
    """Returns accuracy rounded to 4 digits. 
    Predictions are expected to be probabilities. If over #threshold
    prediction is seen as true. 
    """
    pred_integer = (pred > threshold).type(torch.LongTensor)
    accu = (label == pred_integer).sum() / torch.ones(label.shape).sum()
    accu = round(float(accu), 4)

    return accu
