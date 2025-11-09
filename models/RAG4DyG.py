import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from utils.utils import NeighborSampler
from models.modules import TimeDualDecayEncoder

class RAG4DyG(nn.Module):
    """
    A High-Performance Retrieval-Augmented model based on the DyGKT architecture.
    This version includes Layer Normalization and has a corrected function signature
    to match the existing training and evaluation framework.

    Workflow:
    1.  Retrieval: Retrieve student's historical interactions.
    2.  Feature Engineering: Compute rich DyGKT features (node, edge, time, struct).
    3.  Stabilized Fusion: Sum the features and apply LayerNorm.
    4.  Graph Fusion: Construct and process a graph from the history to get a
        fused student knowledge embedding (src_emb).
    5.  Prediction: Use the fused embedding for link prediction.
    """
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim: int = 16,
                 num_neighbors: int = 50,
                 dropout: float = 0.5,
                 device: str = 'cuda:0',
                 **kwargs):

        super(RAG4DyG, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_dim = 64
        self.edge_dim = 64
        self.time_dim = time_dim

        self.projection_layer = nn.ModuleDict({
            'feature_linear': nn.Linear(self.node_raw_features.shape[1], self.node_dim, bias=True),
            'edge': nn.Linear(1, self.node_dim, bias=True),
            'time': nn.Linear(self.time_dim, self.node_dim, bias=True),
            'struct': nn.Linear(1, self.node_dim, bias=True),
        })

        self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)
        self.gnn_fusion = GCNConv(self.node_dim, self.node_dim)
        self.feature_fusion_norm = nn.LayerNorm(self.node_dim)
        self.output_layer = nn.Linear(self.node_dim, self.node_dim, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        
    def get_features(self, nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times):
        node_feats = self.projection_layer['feature_linear'](self.node_raw_features[nodes_neighbor_ids])
        edge_feats = self.projection_layer['edge'](self.edge_raw_features[nodes_edge_ids][:, :, 0].unsqueeze(-1))
        time_feats = self.projection_layer['time'](self.time_encoder(nodes_neighbor_times))
        return node_feats, edge_feats, time_feats

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray = None):
        """
        *** CORRECTED FUNCTION SIGNATURE ***
        This now matches the calling signature from both the training and evaluation loops.
        The unused `retrieved_indices` argument has been removed.
        The `edge_ids` argument is accepted but can be None during evaluation for negative samples.
        """
        batch_size = len(src_node_ids)
        
        # Step 1: Retrieval
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors
        )
        
        src_neighbor_node_ids = torch.from_numpy(src_neighbor_node_ids).long().to(self.device)
        src_neighbor_edge_ids = torch.from_numpy(src_neighbor_edge_ids).long().to(self.device)
        src_neighbor_times = torch.from_numpy(src_neighbor_times).float().to(self.device)
        dst_node_ids_tensor = torch.from_numpy(dst_node_ids).long().to(self.device)

        # Step 2: Feature Engineering
        retrieved_node_feat, retrieved_edge_feat, retrieved_time_feat = self.get_features(
            src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times
        )
        
        co_occurrence_feat = (src_neighbor_node_ids == dst_node_ids_tensor.unsqueeze(1)).float().unsqueeze(-1)
        co_occurrence_emb = self.projection_layer['struct'](co_occurrence_feat)
        
        retrieved_skill_ids = self.node_raw_features[src_neighbor_node_ids][:, :, 0].long()
        current_skill_ids = self.node_raw_features[dst_node_ids_tensor][:, 0].long()
        skill_similarity_feat = (retrieved_skill_ids == current_skill_ids.unsqueeze(1)).float().unsqueeze(-1)
        skill_similarity_emb = self.projection_layer['struct'](skill_similarity_feat)
        
        fused_history_features = (retrieved_node_feat + retrieved_edge_feat + 
                                  retrieved_time_feat + co_occurrence_emb + skill_similarity_emb)
        
        fused_history_features = self.feature_fusion_norm(fused_history_features)

        # Step 3: Graph Construction and Fusion
        graph_data_list = []
        for i in range(batch_size):
            graph_node_features = fused_history_features[i]
            
            num_valid_interactions = (src_neighbor_node_ids[i] > 0).sum()
            if num_valid_interactions < 2:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else:
                nodes_in_graph = torch.arange(num_valid_interactions, device=self.device)
                edge_index = torch.stack([nodes_in_graph[:-1], nodes_in_graph[1:]], dim=0)

            graph_data_list.append(Data(x=graph_node_features, edge_index=edge_index))

        batched_graph = Batch.from_data_list(graph_data_list).to(self.device)
        gnn_output = torch.relu(self.gnn_fusion(batched_graph.x, batched_graph.edge_index))
        src_emb = global_mean_pool(gnn_output, batched_graph.batch)

        # Step 4: Destination Embedding
        dst_node_features = self.node_raw_features[dst_node_ids_tensor]
        dst_emb = self.projection_layer['feature_linear'](dst_node_features)

        src_emb = self.output_layer(src_emb)
        dst_emb = self.output_layer(dst_emb)

        return self.dropout_layer(src_emb), self.dropout_layer(dst_emb)
