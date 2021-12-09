import random
import networkx as nx 
import torch
from torch import nn
import random
import pandas as pd
import numpy as np

def get_mapping(df):
    """
    Map unique items of two groups to the same nodeid domain. 
    (E.g. users get mapped to nodeid from 1 to 123 and movies get mapped to
    nodeid from 124 to 278). 
    
    Params: 
        - df: pandas Dataframe where the first column is group one that 
                is to be mapped and column two is group two that is to be mapped.
                The first column gets mapped to nodeids first then the second group
    Returns: 
        - nodeid_c1: dictionary with keys that are node ids and the items of column 1 
                    as items
        - nodeid_c2: same as nodeid_c1 but with items of column 2
        - c1_nodeid: reverse dictionary of nodeid_c1
        - c2_nodeid: reverse dictionary of nodeid_c2
    
    """
    unique_c1 = df.iloc[:, 0].drop_duplicates().sort_values()
    unique_c2 = df.iloc[:, 1].drop_duplicates().sort_values()

    unique_values = pd.concat([unique_c1, unique_c2], axis=0).reset_index(drop=True)
    nodeid_c1 = unique_values[: len(unique_c1)].to_dict()
    nodeid_c2 = unique_values[len(unique_c1) :].to_dict()
    c1_nodeid = {v: k for k, v in nodeid_c1.items()}
    c2_nodeid = {v: k for k, v in nodeid_c2.items()}

    return nodeid_c1, nodeid_c2, c1_nodeid, c2_nodeid


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
    
    return edge_index


# def sample_negative_edges(G, num_neg_samples):
#     """Returns list of #num_neg_samples of edges that are not in the graph.
#     Edges are represented as tuples. 
#     """
#     neg_edge_list = random.sample(list(nx.non_edges(G)), num_neg_samples)
#     return neg_edge_list


def sample_negative_edges(G, num_neg_samples, threshold=None, seed=25):
    """Random negative edge sampling for undirected graph in O(n_negative_edges)

    Params:
        - G: NX Graph
        - n_neg_edges: number of negative edges to be sampled
        - threshold: If threshold is not none, it is interpreted as the threshold
                splitting a bipartite graph
        - seed: integer random seed for random.seed()

    Returns
        - list of edges, edges are tuples. The edges are undirected

    """
    random.seed(seed)
    result = set()
    nodes = set(G)
    if threshold is not None:
        group1_nodes = set([u for u in nodes if u <= threshold])
        group2_nodes = set([u for u in nodes if u > threshold])
    else:
        group1_nodes = group2_nodes = nodes

    while len(result) < num_neg_samples:

        if len(nodes) == 0:
            raise ValueError("asked for too many negative edges")

        u = random.sample(list(group1_nodes), 1)[0]
        u_nodes = list(group2_nodes - set(G[u]))
        while True:
            if len(u_nodes) == 0:
                group1_nodes.remove(u)
                break
            v = random.sample(u_nodes, 1)[0]
            e1 = (u, v)
            e2 = (v, u)

            if (not e1 in result) & (not e2 in result):
                result.add(e1)
                break
            u_nodes.remove(v)
    return list(result)


def create_node_emb(num_node, embedding_dim=16, random_weights = True):
    """
    Returns nn.Embedding object with #num_node embeddings. 
    Each embedding has dimension of #embedding_dim. 
    Embedding weights are initialized randomly if random weights is True.
    """
    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
    if random_weights: 
        emb.weight.data = torch.rand(emb.weight.data.shape)
    else:
        emb.weight.data = torch.ones(emb.weight.data.shape)/2
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
    accu = round(float(accu), 5)

    return accu


def get_pos_edges_users(pos_edge_index):
    """Takes tensor with positive edges and returns
    Returns:
        - users: user id in each of the edges in pos_edge_index
                shape is (n_edges)
        - unique_users: list of unique user ids
        - index: position of user in pos_edge_index (dim0)
    """
    users, index = pos_edge_index.min(dim=0)
    unique_users = users.unique().tolist()
    return users, unique_users, index


def get_pos_edges_movies(pos_edge_index):
    """Takes tensor with positive edges and returns
    Returns:
        - movies: movie id in each of the edges in pos_edge_index
                shape is (n_edges)
        - unique_users: list of unique movie ids
        - index: position of movie in pos_edge_index (dim0)
    """
    movies, index = pos_edge_index.max(dim=0)
    unique_movies = movies.unique().tolist()
    return movies, unique_movies, index


def user_batch_generator(_unique_users, n_batches=100, batch_n_user=None):
    """Returns a generator object giving a batch of users
    in each step.
    Params:
        - _unique_users: list of unique user ids
        - n_batches: integer of number of batches
        - batch_n_users: integer number of number of users
            in each batch
    """
    unique_user = _unique_users.copy()
    random.shuffle(unique_user)
    if batch_n_user is None:
        batch_n_user = len(unique_user) // n_batches
    for i in range(0, len(unique_user), batch_n_user):
        yield unique_user[i : i + batch_n_user]
        
        
def brp_loss(_f_pos, _f_neg):
    """Return brp loss.

    Params:
        - _f_pos: positive edges for user
        - _f_neg: negative edges for user
    """
    s = nn.Sigmoid()
    f_pos = _f_pos.repeat_interleave(_f_neg.shape[0], dim=0)
    f_neg = _f_neg.repeat(1, _f_pos.shape[0])

    return -torch.log(s(f_pos - f_neg)).mean()


def get_pos_neg_edges_for_user(edges, users, u, unique_movies_set, neg_sample_size=10):
    """ "
    Returns brp loss (float) for a set of users.
    Params:
        - edges: positive edges in graph as tensor, shape is (2, n_edges)
        - users: tensor of shape (n_edges) with user ids in it
        - u: integer user id for which we want to calculate the brp loss (should be in train edges)
        - unique_movies_set: set containing movie ids that appear in positive edges
        - neg_sample_size: size of negative edges for loss calculation
    """

    pos_edges = edges[:, users == u]

    watched_movies = set(pos_edges.max(dim=0)[0].tolist())
    neg_movies = list(unique_movies_set - watched_movies)
    neg_movies = np.random.choice(neg_movies, neg_sample_size, replace=True)
    neg_movies = torch.tensor(neg_movies)

    user_tensor = torch.tensor([u]).repeat(neg_sample_size)
    neg_edges = torch.stack([user_tensor, neg_movies])

    return pos_edges, neg_edges
