from torch import nn
from .graph_transformer_layers import GraphEncoder
from .mlp import MLP
from utils import get_problem_default_node_feat_dim


class GraphTransformer(nn.Module):
    def __init__(self,
                problem: str,
                node_extra_dim: int,
                edge_dim: int,
                hidden_dim: int = 128,
                n_mlp_layers: int = 3,
                n_encoder_layers: int = 8,
                encoder_ffn_embedding_dim: int = 512, 
                dropout: float = 0.1,
                attention_dropout: float = 0.1,
                activation_dropout: float = 0
    ):
        super().__init__()
        
        node_dim = get_problem_default_node_feat_dim(problem) + node_extra_dim

        self.graph_encoder = GraphEncoder(
            num_encoder_layers=n_encoder_layers,
            embedding_dim=hidden_dim,
            ffn_embedding_dim=encoder_ffn_embedding_dim,
            num_attention_heads=4,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
        )

        self.nodes_batchnorm = nn.BatchNorm1d(node_dim, affine=False)
        self.edges_batchnorm = nn.BatchNorm1d(edge_dim, affine=False)
        self.nodes_embedding = nn.Linear(node_dim, hidden_dim, bias=False)
        self.edges_embedding = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.mlp = MLP(hidden_dim, 1024, 2, n_mlp_layers)

    def forward(self, node_feat, edge_feat, edge_index, reachability):
        batch_size = node_feat.size(0)
        node_count = node_feat.size(1)
        edge_count = edge_feat.size(1) // node_count
        node_feat = self.nodes_batchnorm(node_feat.transpose(-1, -2)).transpose(-1, -2)
        edge_feat = self.edges_batchnorm(edge_feat.transpose(-1, -2)).transpose(-1, -2)
        node_embedding = self.nodes_embedding(node_feat)  # batch_size x n_node x hidden_dimension
        edge_embedding = self.edges_embedding(edge_feat).reshape(
            batch_size, node_count, edge_count, -1) # batch_size x n_node x n_edge x hidden_dimension
        x, e = self.graph_encoder(
            node_embedding,
            edge_embedding,
            key_padding_mask=reachability,
            edge_index=edge_index,
        )
        y_edges = self.mlp(e)

        return x.transpose(0, 1), y_edges
