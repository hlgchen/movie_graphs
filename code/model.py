import torch 
from torch import nn


class simple_embedding(): 
    """Simple embedding model where the proximity is just the scalarproduct 
    between the embedding of the nodes. 
    """
    def __init__(self, emb): 
        self.emb = emb
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        
    def forward(self, edges): 
        embedded_nodes = self.emb(edges)
        s_product = torch.mul(embedded_nodes[0], embedded_nodes[1]).sum(axis=1)
        out = self.sigmoid(s_product)
        return out
    
    def recommend(self, edges): 
        """
        Takes possible edges and predicts how likely it is to exist, 
        i.e. estimates how close both nodes in the edge are in terms of 
        their embedding. 
        
        Returns sorted edges (by prediction score) and the sorted prediction score
        """
        pred = self.forward(edges)
        ranking = torch.argsort(pred, descending=True)
        return edges[:, ranking], pred[ranking]
  

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

def gcn_norm(edge_index, edge_weight=None, num_nodes=None): 
    """
    Returns edge index and edge weight that corresonds to edge diffusion
    D^-0.5 A D^-0.5
    
    Params: 
        - edge_index: tensor of shape (2, n_edges) containing edges
        - edge_weight: tensor of shape (n_edges, ) containing weights of each edge
        - num_nodes: integer with number of nodes of the graph
    
    Returns: 
        - edge_index: edge index that was passed to the function 
        - weight: tensor of shape (n_edges, ) containing weights that correspond to diffusion
    
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=torch.float,
                                         device=edge_index.device)
        
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


from torch_geometric.nn.conv import MessagePassing
    
class LightGCNConv(MessagePassing): 
    
    def __init__(self): 
        super(LightGCNConv, self).__init__()
        
    def forward(self, x, edge_index, edge_weight=None): 
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out
    
    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

#     def message_and_aggregate(self, adj_t, x)
#         return matmul(adj_t, x, reduce=self.aggr)
    
    
class LightGCN(torch.nn.Module):
    
    def __init__(self, num_layers):
        
        super(LightGCN, self).__init__()
        
        self.convs = torch.nn.ModuleList([
              LightGCNConv() 
              for i in range(num_layers)
            ])


    def forward(self, x, adj_t, adj_t_is_undirected=True):
        
        """For message parsing edges have to be directed"""
        if adj_t_is_undirected: 
            adj_t_reversed = adj_t[[1,0],:]
            adj_t = torch.cat([adj_t, adj_t_reversed], dim = 1)
        
        out_ls = []
        edge_index, edge_weight = gcn_norm(adj_t, edge_weight=None, num_nodes=x.size(0))
        
        for i in range(len(self.convs)): 
            out_ls.append(x)
            x = self.convs[i](x, edge_index, edge_weight)  
        return torch.stack(out_ls).mean(dim=0)
    
    
    
# class GCN(torch.nn.Module):
    
#     def __init__(self, emb_dim, num_layers, dropout):
        
#         super(GCN, self).__init__()
        
#         self.convs = torch.nn.ModuleList([
#               GCNConv(in_channels = emb_dim, out_channels=emb_dim) 
#               for i in range(num_layers)
#             ])
#         self.bns = torch.nn.ModuleList(
#             [torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers -1)]
#             )
#         # Probability of an element getting zeroed
#         self.dropout = dropout
        

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, x, adj_t):
#         for i in range(len(self.bns)): 
#             x = self.convs[i](x,adj_t)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
        
#         return x