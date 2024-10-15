#I ADDED THIS FILE
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import Linear, Sequential # type: ignore
import torch.nn.functional as F # type: ignore

import torch_geometric
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool # type: ignore

#Altered slightly (e.g., edge weights and self.embedding_size) for compatability with PGExplainer
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.embedding_size = 32 #embedding size for each node
        self.conv1 = GCNConv(input_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim) #normalizes the feature vector
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 2, 2)
        
    
    def embedding(self, x, edge_index, edge_weights=None):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weights))) #x is for one batch; do
        return F.relu(self.bn2(self.conv2(x, edge_index, edge_weights)))

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        
        node_embeddings = self.embedding(x, edge_index, edge_weights)
        
        x_max = global_max_pool(node_embeddings, batch)
        x_mean = global_mean_pool(node_embeddings, batch)
        
        g_embedding = torch.cat([x_max, x_mean], dim=1) #concatenate the embeddings
        
        return self.fc1(g_embedding)