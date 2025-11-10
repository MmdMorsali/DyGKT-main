import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from utils.utils import NeighborSampler
from models.modules import TimeDualDecayEncoder

class RAG4DyG(nn.Module):
    """
    True RAG Version for DyGKT Framework.
    This model is designed to use the pre-computed retriever indices.
    """
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 retrieval_pool_features: dict, # Accepts the pre-computed pool
                 time_dim: int = 8,
                 num_neighbors: int = 100,
                 dropout: float = 0.1,
                 device: str = 'cuda:0',
                 **kwargs):

        super(RAG4DyG, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        # Store the retrieval pool features as tensors on the correct device
        self.retrieved_nodes = torch.from_numpy(retrieval_pool_features['nodes']).long().to(device)
        
        self.node_dim = 32
        self.edge_dim = 32
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
        # This function now needs to handle the extra 'K_RETRIEVE' dimension
        # The input shape will be (batch_size, K, num_neighbors)
        node_feats = self.projection_layer['feature_linear'](self.node_raw_features[nodes_neighbor_ids])
        
        # We can't use edge features from the pool easily, so we zero them out for demonstrations
        # A more complex implementation could store these too, but this is a robust start.
        edge_feats = torch.zeros_like(node_feats)

        # We also cannot easily get time features, so we use the student's own history for that
        time_feats = self.projection_layer['time'](self.time_encoder(nodes_neighbor_times))
        return node_feats, edge_feats, time_feats

    # The signature now accepts 'retrieved_indices'
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 retrieved_indices: np.ndarray,
                                                 edge_ids: np.ndarray = None):
        
        batch_size = len(src_node_ids)
        dst_node_ids_tensor = torch.from_numpy(dst_node_ids).long().to(self.device)
        retrieved_indices = torch.from_numpy(retrieved_indices).long().to(self.device)
        
        # --- Step 1: Use pre-computed indices to retrieve demonstrations from the pool ---
        # Shape: (batch_size, K_RETRIEVE, num_neighbors)
        demonstration_nodes = self.retrieved_nodes[retrieved_indices]

        # We need features for these demonstrations.
        # This part is simplified: we only use node features from the demos.
        # Shape: (batch_size, K_RETRIEVE, num_neighbors, node_dim)
        demo_node_feat = self.projection_layer['feature_linear'](self.node_raw_features[demonstration_nodes])
        
        # --- Step 2: Feature Engineering for Demonstrations ---
        # Reshape for broadcasting: (batch_size, 1, 1, node_dim)
        current_skill_ids = self.node_raw_features[dst_node_ids_tensor][:, 0].long().view(batch_size, 1, 1)
        demo_skill_ids = self.node_raw_features[demonstration_nodes][..., 0].long()
        
        skill_sim_feat = (demo_skill_ids == current_skill_ids).float().unsqueeze(-1)
        skill_sim_emb = self.projection_layer['struct'](skill_sim_feat)
        
        # Combine features for demonstrations
        fused_demo_features = demo_node_feat + skill_sim_emb
        
        # --- Step 3: Fuse the K demonstrations into one history ---
        # We average the features across the K demonstrations.
        # Shape: (batch_size, num_neighbors, node_dim)
        fused_history_features = fused_demo_features.mean(dim=1)
        
        # Normalize the fused features
        fused_history_features = self.feature_fusion_norm(fused_history_features)

        # --- Step 4: Graph Construction and Student Embedding ---
        graph_data_list = []
        for i in range(batch_size):
            num_valid_interactions = (demonstration_nodes[i, 0] > 0).sum()
            if num_valid_interactions < 2:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else:
                nodes_in_graph = torch.arange(num_valid_interactions, device=self.device)
                edge_index = torch.stack([nodes_in_graph[:-1], nodes_in_graph[1:]], dim=0)

            graph_data_list.append(Data(x=fused_history_features[i], edge_index=edge_index))

        batched_graph = Batch.from_data_list(graph_data_list).to(self.device)
        gnn_output = torch.relu(self.gnn_fusion(batched_graph.x, batched_graph.edge_index))
        src_emb = global_mean_pool(gnn_output, batched_graph.batch)

        # --- Step 5: Destination (Question) Embedding ---
        dst_node_features = self.node_raw_features[dst_node_ids_tensor]
        dst_emb = self.projection_layer['feature_linear'](dst_node_features)

        src_emb = self.output_layer(src_emb)
        dst_emb = self.output_layer(dst_emb)

        return self.dropout_layer(src_emb), self.dropout_layer(dst_emb)


