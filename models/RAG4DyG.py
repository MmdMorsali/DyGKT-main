%%writefile models/RAG4DyG.py
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from utils.utils import NeighborSampler
from models.modules import TimeDualDecayEncoder

class RAG4DyG(nn.Module):
    """
    A Corrected and Simplified Retrieval-Augmented Generation model for Knowledge Tracing,
    designed to integrate with the DyGKT benchmarking framework.

    This version performs on-the-fly retrieval and fusion for each batch.
    1.  Retrieval: For each student, retrieve their own recent historical interactions.
    2.  Fusion: Construct a summary graph from the retrieved history and use a GNN
        to compute a single 'fused_embedding' representing the student's knowledge state.
    3.  Prediction: Use this fused embedding as the source embedding for the link prediction task.
    """
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 num_neighbors: int = 50,
                 time_dim: int = 16,
                 dropout: float = 0.5,
                 device: str = 'cuda:0',
                 **kwargs):

        super(RAG4DyG, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device

        # Move raw features to the specified device
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        # Define dimensions
        self.node_dim = 64
        self.edge_dim = 64
        self.time_dim = time_dim

        # Projection layers (inspired by DyGKT)
        self.projection_layer = nn.ModuleDict({
            'feature_linear': nn.Linear(self.node_raw_features.shape[1], self.node_dim, bias=True),
            'edge': nn.Linear(self.edge_raw_features.shape[1], self.edge_dim, bias=True),
        })

        # Time encoder
        self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)
        
        # --- Graph Fusion Module (GNN) ---
        # This will process the graph constructed from retrieved interactions.
        self.gnn_fusion = GCNConv(self.node_dim, self.node_dim)

        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.node_dim, self.node_dim, bias=True)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """Set the neighbor sampler to handle the 'Retrieval' step."""
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 dst_node_ids: np.ndarray):
        """
        Computes retrieval-augmented student embeddings and standard question embeddings.
        """
        # --- Step 1: Retrieval ---
        # Retrieve historical interactions for the source nodes (students).
        retrieved_node_ids, retrieved_edge_ids, retrieved_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=self.num_neighbors
        )
        
        # Convert numpy arrays to tensors
        retrieved_node_ids = torch.from_numpy(retrieved_node_ids).long().to(self.device)
        retrieved_edge_ids = torch.from_numpy(retrieved_edge_ids).long().to(self.device)
        
        # --- Step 2: Graph Construction and Fusion ---
        batch_size = len(src_node_ids)
        graph_data_list = []

        for i in range(batch_size):
            # Get the unique nodes and their original IDs from the retrieved history
            unique_nodes, unique_indices = torch.unique(retrieved_node_ids[i], return_inverse=True)
            
            # Get features for these unique nodes
            node_features = self.node_raw_features[unique_nodes]
            projected_node_features = self.projection_layer['feature_linear'](node_features)

            # Create edge index for the graph using the unique indices
            # An interaction (u, v) is an edge. Here we link consecutive nodes in the history.
            # We create a simple sequential graph for each student's history
            num_valid_edges = (retrieved_node_ids[i] > 0).sum()
            if num_valid_edges < 2: # Not enough nodes to form an edge
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else:
                valid_indices = unique_indices[:num_valid_edges]
                edge_index = torch.stack([valid_indices[:-1], valid_indices[1:]], dim=0)

            graph_data_list.append(Data(x=projected_node_features, edge_index=edge_index))

        # Create a single batched graph for efficient GNN processing
        batched_graph = Batch.from_data_list(graph_data_list).to(self.device)

        # Process the batched graph with the GNN
        gnn_output = self.gnn_fusion(batched_graph.x, batched_graph.edge_index)
        gnn_output = torch.relu(gnn_output)
        
        # Use global mean pooling to get a single vector per student history
        src_emb = global_mean_pool(gnn_output, batched_graph.batch)

        # --- Step 3: Get Current Question (Destination) Embedding ---
        current_question_ids = torch.from_numpy(dst_node_ids).long().to(self.device)
        dst_node_features = self.node_raw_features[current_question_ids]
        dst_emb = self.projection_layer['feature_linear'](dst_node_features)

        # Apply final layers
        src_emb = self.output_layer(src_emb)
        dst_emb = self.output_layer(dst_emb)

        return self.dropout_layer(src_emb), self.dropout_layer(dst_emb)
