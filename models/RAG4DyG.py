import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils.utils import NeighborSampler
from models.modules import TimeDualDecayEncoder

class RAG4DyG(nn.Module):
    """
    An improved Retrieval-Augmented Generation model for Knowledge Tracing.
    This version integrates feature engineering from DyGKT into the RAG framework.
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
            'struct': nn.Linear(1, self.node_dim, bias=True),
        })

        # Time encoder (using the same as DyGKT)
        self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)

        # --- Use GCN instead of GAT for static graph support ---
        self.gcn_layer = GCNConv(self.node_dim, self.node_dim)  # Use GCNConv here for static graphs
        
        self.output_layer = nn.Linear(self.node_dim, self.node_dim, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """Set the neighbor sampler to handle the 'Retrieval' step."""
        self.neighbor_sampler = neighbor_sampler

    def time_decay(self, query_time, candidate_time, lambda_decay=0.1):
        """
        Time-decay function to prioritize temporally relevant samples.
        """
        return torch.exp(-lambda_decay * torch.abs(query_time - candidate_time))

    def contrastive_loss(self, query_emb, positive_emb, negative_emb, temperature=0.1):
        """
        Contrastive loss to help the model differentiate between positive and negative examples.
        """
        pos_sim = F.cosine_similarity(query_emb, positive_emb)
        neg_sim = F.cosine_similarity(query_emb, negative_emb)
        loss = -torch.log(torch.exp(pos_sim / temperature) / (torch.exp(pos_sim / temperature) + torch.exp(neg_sim / temperature)))
        return loss

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
        
        # --- Step 3: Fusion using Graph Convolution (GCN) ---
        
        # Combine all features for each item in the retrieved sequence
        fused_interaction_sequence = (retrieved_node_emb + 
                                      retrieved_edge_emb + 
                                      retrieved_time_emb + 
                                      co_occurrence_emb + 
                                      skill_similarity_emb)

        # Apply GCN for richer interaction modeling
        edge_index = torch.arange(fused_interaction_sequence.size(0)).unsqueeze(0).repeat(2, 1)  # Creating dummy edges for GCN
        gcn_out = self.gcn_layer(fused_interaction_sequence, edge_index)  # Pass to GCNConv layer
        
        # The final hidden state `src_emb` is the retrieval-augmented student embedding.
        src_emb = gcn_out.mean(dim=1)  # or gcn_out.squeeze(0) depending on your requirements
        
        # --- Step 4: Get Current Question Embedding ---
        dst_node_features = self.node_raw_features[current_question_ids]
        dst_emb = self.projection_layer['feature_linear'](dst_node_features)

        # Apply final layers
        src_emb = self.output_layer(src_emb)
        dst_emb = self.output_layer(dst_emb)

        # Contrastive learning loss: Query is src_emb, positive is dst_emb, negative samples from retrieval
        negative_emb = self.node_raw_features[retrieved_node_ids[torch.randint(0, len(retrieved_node_ids), (src_emb.size(0),))]]
        loss = self.contrastive_loss(src_emb, dst_emb, negative_emb)

        return self.dropout_layer(src_emb), self.dropout_layer(dst_emb), loss
