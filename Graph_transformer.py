import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class NodeEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(NodeEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
    
    def forward(self, x):
        return self.embedding(x)

class EdgeEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(EdgeEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
    
    def forward(self, edge_attr):
        return self.embedding(edge_attr)

class GraphPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super(GraphPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
    
    def forward(self, x, edge_index):
        # Compute positional encodings (e.g., based on graph Laplacian)
        # Here, we can use predefined methods or libraries
        pass

class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GraphTransformerLayer, self).__init__()
        self.attention = pyg_nn.GATConv(embed_dim, embed_dim // num_heads, heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.attention(x, edge_index, edge_attr)
        h = self.layer_norm(h + x)  # Residual connection and normalization
        h = self.feed_forward(h)
        return h
    
class ReadoutLayer(nn.Module):
    def __init__(self, embed_dim):
        super(ReadoutLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)  # Predict some property
        )
    
    def forward(self, x):
        return torch.mean(x, dim=0)  # Aggregate node features
    
class GraphTransformer(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, embed_dim, num_heads, num_layers):
        super(GraphTransformer, self).__init__()
        self.node_embedding = NodeEmbedding(node_input_dim, embed_dim)
        self.edge_embedding = EdgeEmbedding(edge_input_dim, embed_dim)
        self.positional_encoding = GraphPositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([GraphTransformerLayer(embed_dim, num_heads) for _ in range(num_layers)])
        self.readout = ReadoutLayer(embed_dim)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        x = self.positional_encoding(x, edge_index)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        return self.readout(x)


def build_graph_data(atom_features, bond_features, edge_index):
    node_features = torch.tensor(atom_features, dtype=torch.float)
    edge_features = torch.tensor(bond_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    return Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)

# Example usage
atom_features = [[...], [...]]  # List of atom features
bond_features = [[...], [...]]  # List of bond features
edge_index = [[0, 1], [1, 0]]  # List of edges (source, target)

graph_data = build_graph_data(atom_features, bond_features, edge_index)
