import numpy as np
import torch
import torch.nn as nn

from utils.utils import NeighborSampler
from models.modules import TimeDualDecayEncoder

class RAG4DyG(nn.Module):
    """
    Corrected RAG-Augmented DyGKT Model.

    This version performs true augmentation by combining the student's own history
    (processed via DyGKT's methods) with a context vector derived from
    retrieved demonstrations.
    """
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 retrieval_pool_features: dict,
                 time_dim: int = 16,
                 num_neighbors: int = 100,
                 dropout: float = 0.1,
                 device: str = 'cuda:0',
                 **kwargs):

        super(RAG4DyG, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device

        # --- Core DyGKT components ---
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_dim = 64  # Standard dimension for DyGKT
        self.time_dim = time_dim

        self.projection_layer = nn.ModuleDict({
            'feature_linear': nn.Linear(self.node_raw_features.shape[1], self.node_dim, bias=True),
            'edge': nn.Linear(1, self.node_dim, bias=True),
            'time': nn.Linear(self.time_dim, self.node_dim, bias=True),
            'struct': nn.Linear(1, self.node_dim, bias=True),
        })

        self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)
        self.student_updater = nn.GRU(input_size=self.node_dim, hidden_size=self.node_dim, batch_first=True)
        self.question_updater = nn.GRU(input_size=self.node_dim, hidden_size=self.node_dim, batch_first=True)

        # --- RAG components ---
        self.retrieved_nodes = torch.from_numpy(retrieval_pool_features['nodes']).long().to(device)
        self.rag_context_dim = 32  # Dimension for the retrieved context vector
        
        self.demo_fusion_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.rag_context_dim)
        )
        self.demo_fusion_norm = nn.LayerNorm(self.rag_context_dim)

        # --- Final Fusion & Output Layers ---
        final_student_dim = self.node_dim + self.rag_context_dim
        self.student_output_layer = nn.Linear(final_student_dim, self.node_dim)
        self.question_output_layer = nn.Linear(self.node_dim, self.node_dim) # For the question embedding
        self.dropout_layer = nn.Dropout(dropout)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def _get_history_features(self, node_ids, interact_times, num_hist_neighbors):
        """Helper to get features for a given set of nodes and their histories."""
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids, interact_times, num_hist_neighbors
        )

        node_feats = self.projection_layer['feature_linear'](self.node_raw_features[neighbor_node_ids])
        edge_feats = self.projection_layer['edge'](self.edge_raw_features[neighbor_edge_ids][..., 0].unsqueeze(-1))
        time_feats = self.projection_layer['time'](self.time_encoder(torch.from_numpy(neighbor_times).float().to(self.device)))

        return node_feats, edge_feats, time_feats, torch.from_numpy(neighbor_node_ids).to(self.device)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 retrieved_indices: np.ndarray,
                                                 edge_ids: np.ndarray):
        
        # =================================================================================
        # Part 1: Process the student's and question's OWN history (The DyGKT Way)
        # =================================================================================
        
        # --- Student (Source) Base Embedding ---
        student_node_feats, student_edge_feats, student_time_feats, student_neighbor_ids = self._get_history_features(
            src_node_ids, node_interact_times, self.num_neighbors
        )
        
        # --- Add Multiset Indicator (Critical DyGKT Feature) ---
        dst_node_ids_t = torch.from_numpy(dst_node_ids).long().to(self.device)
        # Check if current question (dst_node) appeared in student's history
        question_multiset = (student_neighbor_ids == dst_node_ids_t.unsqueeze(1)).float().unsqueeze(-1)
        # Check if concept of current question appeared in student's history
        current_concept_ids = self.node_raw_features[dst_node_ids_t][:, 0].long()
        historical_concept_ids = self.node_raw_features[student_neighbor_ids][..., 0].long()
        concept_multiset = (historical_concept_ids == current_concept_ids.unsqueeze(1)).float().unsqueeze(-1)
        
        struct_feats = self.projection_layer['struct'](question_multiset + concept_multiset)
        
        # Combine all features for the student's history
        student_history_combined = student_node_feats + student_edge_feats + student_time_feats + struct_feats
        _, student_base_emb = self.student_updater(student_history_combined)
        student_base_emb = student_base_emb.squeeze(0)

        # --- Question (Destination) Base Embedding ---
        question_node_feats, question_edge_feats, question_time_feats, _ = self._get_history_features(
            dst_node_ids, node_interact_times, self.num_neighbors
        )
        question_history_combined = question_node_feats + question_edge_feats + question_time_feats
        _, question_base_emb = self.question_updater(question_history_combined)
        question_base_emb = question_base_emb.squeeze(0)

        # =================================================================================
        # Part 2: Process the RETRIEVED demonstrations (The RAG Way)
        # =================================================================================
        
        retrieved_indices_t = torch.from_numpy(retrieved_indices).long().to(self.device)
        # Shape: (batch_size, K_RETRIEVE, num_neighbors)
        demonstration_nodes = self.retrieved_nodes[retrieved_indices_t]
        
        # Get features for these demonstrations
        # Shape: (batch_size, K_RETRIEVE, num_neighbors, node_dim)
        demo_node_feat = self.projection_layer['feature_linear'](self.node_raw_features[demonstration_nodes])
        
        # Simple mean pooling over the neighbors in each demonstration
        # Shape: (batch_size, K_RETRIEVE, node_dim)
        demo_summary_feat = demo_node_feat.mean(dim=2)
        
        # Fuse the K demonstrations into a single context vector for each student
        # Shape: (batch_size, rag_context_dim)
        retrieved_context_emb = self.demo_fusion_mlp(demo_summary_feat.mean(dim=1))
        retrieved_context_emb = self.demo_fusion_norm(retrieved_context_emb)

        # =================================================================================
        # Part 3: AUGMENT the student's state with the retrieved context
        # =================================================================================
        
        # Concatenate the student's own state with the context from RAG
        # Shape: (batch_size, node_dim + rag_context_dim)
        augmented_student_emb = torch.cat([student_base_emb, retrieved_context_emb], dim=1)
        
        # Project back to the standard dimension
        final_student_emb = self.student_output_layer(augmented_student_emb)
        final_question_emb = self.question_output_layer(question_base_emb)

        return self.dropout_layer(final_student_emb), self.dropout_layer(final_question_emb)
