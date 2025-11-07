import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from models.modules import TimeDualDecayEncoder

class RAG4DyG(nn.Module):
    """
    An improved Retrieval-Augmented Generation model for Knowledge Tracing.

    This version is a hybrid that integrates the powerful feature engineering from DyGKT 
    into the RAG sequence-processing framework.
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
        
        # Move raw features to the specified device during initialization
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        # Define dimensions (aligned with DyGKT for compatibility)
        self.node_dim = 64
        self.edge_dim = 64
        self.time_dim = time_dim

        # --- Borrowing projection layers from DyGKT ---
        self.projection_layer = nn.ModuleDict({
            'feature_linear': nn.Linear(self.node_raw_features.shape[1], self.node_dim, bias=True),
            'edge': nn.Linear(self.edge_raw_features.shape[1], self.edge_dim, bias=True),
            'time': nn.Linear(self.time_dim, self.node_dim, bias=True),
            # This 'struct' layer is crucial for modeling co-occurrence and skill similarity
            'struct': nn.Linear(1, self.node_dim, bias=True),
        })

        # Time encoder (using the same as DyGKT)
        self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)

        # The GRU module for fusing the augmented sequence of interactions
        self.retrieval_fusion_module = nn.GRU(
            input_size=self.node_dim, # Input will be the combined feature vector for each historical interaction
            hidden_size=self.node_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.node_dim, self.node_dim, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """Set the neighbor sampler to handle the 'Retrieval' step."""
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 dst_node_ids: np.ndarray):
        """
        Computes retrieval-augmented student embeddings and standard question embeddings.
        This version incorporates DyGKT's feature engineering.
        """
        
        # --- Step 1: Retrieval ---
        # Retrieve historical interactions for the source nodes (students).
        retrieved_node_ids, retrieved_edge_ids, retrieved_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=self.num_neighbors
        )
        
        # Convert numpy arrays to tensors on the correct device
        retrieved_node_ids = torch.from_numpy(retrieved_node_ids).long().to(self.device)
        retrieved_edge_ids = torch.from_numpy(retrieved_edge_ids).long().to(self.device)
        retrieved_times = torch.from_numpy(retrieved_times).float().to(self.device)
        
        # Get features for the current questions (destination nodes)
        current_question_ids = torch.from_numpy(dst_node_ids).long().to(self.device)
        current_question_skills = self.node_raw_features[current_question_ids][:, 0].long()

        # --- Step 2: Feature Engineering (Inspired by DyGKT) ---

        # 2a. Base Embeddings for the retrieved sequence
        retrieved_node_features = self.node_raw_features[retrieved_node_ids]
        retrieved_edge_features = self.edge_raw_features[retrieved_edge_ids]

        retrieved_node_emb = self.projection_layer['feature_linear'](retrieved_node_features)
        retrieved_edge_emb = self.projection_layer['edge'](retrieved_edge_features)
        
        time_features = self.time_encoder(retrieved_times)
        retrieved_time_emb = self.projection_layer['time'](time_features)

        # 2b. Structural Co-occurrence Features
        co_occurrence_feat = (retrieved_node_ids == current_question_ids.unsqueeze(1)).float().unsqueeze(-1)
        co_occurrence_emb = self.projection_layer['struct'](co_occurrence_feat)
        
        # 2c. Skill Similarity Features
        retrieved_skill_ids = retrieved_node_features[:, :, 0].long()
        skill_similarity_feat = (retrieved_skill_ids == current_question_skills.unsqueeze(1)).float().unsqueeze(-1)
        skill_similarity_emb = self.projection_layer['struct'](skill_similarity_feat)
        
        # --- Step 3: Fusion ---
        
        # Combine all features for each item in the retrieved sequence
        # This is the key step that gives the model rich context for every historical interaction
        fused_interaction_sequence = (retrieved_node_emb + 
                                      retrieved_edge_emb + 
                                      retrieved_time_emb + 
                                      co_occurrence_emb + 
                                      skill_similarity_emb)

        # The GRU processes the sequence to produce a fused student state.
        # The final hidden state `src_emb` is the retrieval-augmented student embedding.
        _, src_emb = self.retrieval_fusion_module(fused_interaction_sequence)
        src_emb = src_emb.squeeze(0)
        
        # --- Step 4: Get Current Question Embedding ---
        # Project the features of the current destination node (question).
        dst_node_features = self.node_raw_features[current_question_ids]
        dst_emb = self.projection_layer['feature_linear'](dst_node_features)

        # Apply final layers
        src_emb = self.output_layer(src_emb)
        dst_emb = self.output_layer(dst_emb)

        return self.dropout_layer(src_emb), self.dropout_layer(dst_emb)
