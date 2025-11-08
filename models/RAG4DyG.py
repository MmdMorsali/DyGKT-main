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

    This model combines the powerful feature engineering of DyGKT with the graph fusion
    concept from the RAG4DyG paper.

    Workflow:
    1.  **Retrieval**: For each student (src_node), retrieve their historical interactions
        using the same NeighborSampler as DyGKT.
    2.  **Feature Engineering**: Compute the rich features for this history (node, edge,
        time, co-occurrence, skill similarity) exactly as DyGKT does.
    3.  **Graph Fusion**: Instead of using a GRU, construct a dynamic graph from the
        retrieved history. A GNN then processes this graph to produce a single,
        fused embedding representing the student's knowledge state (src_emb).
    4.  **Prediction**: The final prediction is made using this fused student embedding
        and the target question embedding (dst_emb).
    """
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim: int = 16,
                 num_neighbors: int = 100,
                 dropout: float = 0.1,
                 device: str = 'cuda:0',
                 **kwargs):

        super(RAG4DyG, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device

        # --- Inherit Core Components from DyGKT ---
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_dim = 64
        self.edge_dim = 64
        self.time_dim = time_dim

        # Use the same projection layers as DyGKT
        self.projection_layer = nn.ModuleDict({
            'feature_linear': nn.Linear(self.node_raw_features.shape[1], self.node_dim, bias=True),
            'edge': nn.Linear(1, self.node_dim, bias=True),
            'time': nn.Linear(self.time_dim, self.node_dim, bias=True),
            'struct': nn.Linear(1, self.node_dim, bias=True),
        })

        # Use the same time encoder as DyGKT
        self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)
        
        # --- RAG Component: Replace GRU with a GNN Fusion Module ---
        # This GNN will process the graph constructed from the retrieved history.
        self.gnn_fusion = GCNConv(self.node_dim, self.node_dim)

        self.output_layer = nn.Linear(self.node_dim, self.node_dim, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """Set the neighbor sampler to handle the 'Retrieval' step."""
        self.neighbor_sampler = neighbor_sampler
        
    def get_features(self, nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times, node_interact_times):
        """Helper function to compute features, adapted from DyGKT."""
        # Node features
        node_feats = self.projection_layer['feature_linear'](self.node_raw_features[nodes_neighbor_ids])
        
        # Edge features
        edge_feats = self.projection_layer['edge'](self.edge_raw_features[nodes_edge_ids][:, :, 0].unsqueeze(-1))
        
        # Time features
        time_feats = self.time_encoder(nodes_neighbor_times)
        time_feats = self.projection_layer['time'](time_feats)
        
        return node_feats, edge_feats, time_feats

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 dst_node_ids: np.ndarray):
        """
        Computes retrieval-augmented student embeddings and standard question embeddings.
        """
        batch_size = len(src_node_ids)
        
        # --- Step 1: Retrieval ---
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors
        )
        
        # Convert to tensors
        src_neighbor_node_ids = torch.from_numpy(src_neighbor_node_ids).long().to(self.device)
        src_neighbor_edge_ids = torch.from_numpy(src_neighbor_edge_ids).long().to(self.device)
        src_neighbor_times = torch.from_numpy(src_neighbor_times).float().to(self.device)
        
        dst_node_ids_tensor = torch.from_numpy(dst_node_ids).long().to(self.device)

        # --- Step 2: Feature Engineering (from DyGKT) ---
        # Get base features for the retrieved sequence
        retrieved_node_feat, retrieved_edge_feat, retrieved_time_feat = self.get_features(
            src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times, torch.from_numpy(node_interact_times).float().to(self.device)
        )
        
        # Co-occurrence features
        co_occurrence_feat = (src_neighbor_node_ids == dst_node_ids_tensor.unsqueeze(1)).float().unsqueeze(-1)
        co_occurrence_emb = self.projection_layer['struct'](co_occurrence_feat)
        
        # Skill similarity features
        retrieved_skill_ids = self.node_raw_features[src_neighbor_node_ids][:, :, 0].long()
        current_skill_ids = self.node_raw_features[dst_node_ids_tensor][:, 0].long()
        skill_similarity_feat = (retrieved_skill_ids == current_skill_ids.unsqueeze(1)).float().unsqueeze(-1)
        skill_similarity_emb = self.projection_layer['struct'](skill_similarity_feat)
        
        # Combine all features for each item in the retrieved history
        fused_history_features = (retrieved_node_feat + retrieved_edge_feat + 
                                  retrieved_time_feat + co_occurrence_emb + skill_similarity_emb)

        # --- Step 3: Graph Construction and Fusion ---
        graph_data_list = []
        for i in range(batch_size):
            # Features for the nodes in the graph are the fused history features
            graph_node_features = fused_history_features[i]
            
            # Create a simple sequential graph (path graph) from the history
            num_valid_interactions = (src_neighbor_node_ids[i] > 0).sum()
            if num_valid_interactions < 2:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else:
                nodes_in_graph = torch.arange(num_valid_interactions, device=self.device)
                edge_index = torch.stack([nodes_in_graph[:-1], nodes_in_graph[1:]], dim=0)

            graph_data_list.append(Data(x=graph_node_features, edge_index=edge_index))

        # Create a single batched graph for efficient processing
        batched_graph = Batch.from_data_list(graph_data_list).to(self.device)

        # Process with GNN
        gnn_output = self.gnn_fusion(batched_graph.x, batched_graph.edge_index)
        gnn_output = torch.relu(gnn_output)
        
        # Pool the node embeddings to get a single vector for each student's history
        src_emb = global_mean_pool(gnn_output, batched_graph.batch)

        # --- Step 4: Get Current Question (Destination) Embedding ---
        dst_node_features = self.node_raw_features[dst_node_ids_tensor]
        dst_emb = self.projection_layer['feature_linear'](dst_node_features)

        # Apply final layers
        src_emb = self.output_layer(src_emb)
        dst_emb = self.output_layer(dst_emb)

        return self.dropout_layer(src_emb), self.dropout_layer(dst_emb)

