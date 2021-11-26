import networkx as nx 
import torch
from torch import nn
import random

def graph_to_edge_list(G):
    return [edge for edge in nx.edges(G)]

def edge_list_to_tensor(edge_list):
    edge_index = torch.tensor(edge_list)
    edge_index = torch.transpose(edge_index, 0,1)
    return edge_index


def sample_negative_edges(G, num_neg_samples):
    neg_edge_list = random.sample(list(nx.non_edges(G)), num_neg_samples)
    return neg_edge_list


def create_node_emb(num_node, embedding_dim=16):
    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
    emb.weight.data = torch.rand(emb.weight.data.shape)
    return emb


def transductive_edge_split(edge_list, 
                            split_dict,
                            seed=25
                           ): 
    random.seed(seed)
    
    num_edges = len(edge_list)
    idx = list(range(num_edges))
    random.shuffle(idx)
    
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
