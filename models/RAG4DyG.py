import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler

class RAG4DyG(nn.Module):
    """
    Retrieval-Augmented Generation for Dynamic Graphs in Knowledge Tracing.

    This model adapts the RAG concept to the link prediction framework for KT.
    1. Retrieval: Uses the NeighborSampler to get a student's historical interactions.
    2. Fusion: A GRU fuses the sequence of retrieved interactions into a single state vector.
    3. Prediction: The final embeddings are passed to the standard MergeLayer.
    """
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 num_neighbors: int = 50,
                 dropout: float = 0.5,
                 device: str = 'cuda:0',
                 **kwargs):  # Absorb any other unused arguments
        
        super(RAG4DyG, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device
        
        # Initialize raw feature tensors on the correct device
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        # Define dimensions (aligned with DyGKT for compatibility)
        self.node_dim = 64
        self.edge_feature_dim = self.edge_raw_features.shape[1]

        # Projection layers to create embeddings from raw features
        self.node_feature_projection = nn.Linear(self.node_raw_features.shape[1], self.node_dim, bias=True)
        self.edge_feature_projection = nn.Linear(self.edge_feature_dim, self.node_dim, bias=True)

        # The core of the RAG adaptation: a GRU to fuse the retrieved sequence.
        # The input to the GRU is the concatenation of a historical question and its outcome.
        self.fusion_input_dim = self.node_dim + self.node_dim  # hist_question_emb + hist_outcome_emb
        
        self.retrieval_fusion_module = nn.GRU(
            input_size=self.fusion_input_dim,
            hidden_size=self.node_dim,
            batch_first=True
        )

        self.dropout_layer = nn.Dropout(dropout)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        Set the neighbor sampler, which handles the 'Retrieval' step.
        """
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 dst_node_ids: np.ndarray):
        """
        Computes retrieval-augmented student embeddings and standard question embeddings.
        """
        
        # --- Step 1: Retrieval ---
        # Retrieve historical interactions (neighbors, edges, times) for the source nodes (students).
        retrieved_neighbor_nodes, retrieved_edge_ids, _ = self.neighbor_sampler.get_historical_neighbors(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=self.num_neighbors
        )

        # --- Step 2: Prepare Features for Fusion ---
        
        # Get raw features for the retrieved historical questions and outcomes.
        retrieved_node_features = self.node_raw_features[torch.from_numpy(retrieved_neighbor_nodes).long()]
        retrieved_edge_features = self.edge_raw_features[torch.from_numpy(retrieved_edge_ids).long()]
        
        # Project raw features into the common embedding space.
        retrieved_node_embeddings = self.node_feature_projection(retrieved_node_features)
        retrieved_edge_embeddings = self.edge_feature_projection(retrieved_edge_features)

        # Concatenate to form a sequence of 'interaction' embeddings.
        interaction_sequence_embeddings = torch.cat([retrieved_node_embeddings, retrieved_edge_embeddings], dim=-1)

        # --- Step 3: Fusion ---
        # The GRU processes the sequence to produce a fused student state.
        # The final hidden state `src_emb` is the retrieval-augmented student embedding.
        _, src_emb = self.retrieval_fusion_module(interaction_sequence_embeddings)
        src_emb = src_emb.squeeze(0)

        # --- Step 4: Get Current Question Embedding ---
        # Project the features of the current destination node (question).
        dst_node_features = self.node_raw_features[torch.from_numpy(dst_node_ids).long()]
        dst_emb = self.node_feature_projection(dst_node_features)

        return self.dropout_layer(src_emb), self.dropout_layer(dst_emb)