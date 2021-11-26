import torch 
from torch import nn


class simple_embedding(): 
    
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
        pred = self.forward(edges)
        ranking = torch.argsort(pred, descending=True)
        return edges[:, ranking], pred[ranking]
  

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, add_self_loop=False): 
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                         device=edge_index.device)
    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


# from torch_geometric.nn.conv import MessagePassing
    
# class LightGCNConv(MessagePassing): 
    
#     def __init__(self, emb_dimension: int): 
#         self.emb_dimension = emb_dimension
        
#     def forward(self, x, edge_index, edge_weight): 
#         edge_index = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
#                               self.improved, self.add_self_loops)
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
#                              size=None)
#         return out
    
#     def message(self, x_j, edge_weight):
#         return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

# #     def message_and_aggregate(self, adj_t, x)
# #         return matmul(adj_t, x, reduce=self.aggr)
    
    
# class LightGCN(torch.nn.Module):
    
#     def __init__(self, emb_dim, num_layers):
        
#         # super(GCN, self).__init__()
        
#         self.convs = torch.nn.ModuleList([
#               LightGCNConv(emb_dimension = emb_dim) 
#               for i in range(num_layers)
#             ])


#     def forward(self, x, adj_t):
#         out_ls = []
        
#         for i in range(len(self.convs)): 
#             out_ls.append(x)
#             x = self.convs[i](x,adj_t)  
#         return torch.stack(out_ls).mean(dim=0)
    
    
    
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