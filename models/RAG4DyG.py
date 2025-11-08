# --- Full RAG4DyG Model Definition (True Retrieval-Augmented Version) ---

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import networkx as nx
from torch_geometric.utils import from_networkx

from utils.utils import NeighborSampler
# We assume the modules file is in models/modules.py
from models.modules import TimeDualDecayEncoder

# Helper class from original DyGKT that the RAG model uses for its local path
class DyKT_Seq(nn.Module):
    def __init__(self, edge_dim: int, node_dim: int):
        super(DyKT_Seq, self).__init__()
        self.hid_node_updater = nn.GRU(input_size=edge_dim, hidden_size=node_dim, batch_first=True)

    def update(self, x):
        _, hidden = self.hid_node_updater(x)
        return torch.squeeze(hidden, dim=0)

class RAG4DyG(nn.Module):
    """
    A true Retrieval-Augmented Generation model for Knowledge Tracing,
    combining the strengths of DyGKT and the RAG4DyG framework.
    """
    def __init__(self,
                 node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 retrieval_pool_features: dict, # New requirement!
                 time_dim: int = 16,
                 num_neighbors: int = 50,
                 dropout: float = 0.2,
                 device: str = 'cuda:0',
                 **kwargs):

        super(RAG4DyG, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device
        self.node_dim = 64
        self.time_dim = time_dim

        # --- Feature Storage ---
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        
        # The retrieval pool contains all historical sequences from the training set.
        self.retrieval_pool_nodes = torch.from_numpy(retrieval_pool_features['nodes']).long().to(device)
        
        # --- DyGKT Projection & Encoders ---
        self.projection_layer = nn.ModuleDict({
            'feature_linear': nn.Linear(self.node_raw_features.shape[1], self.node_dim, bias=True),
            'edge': nn.Linear(1, self.node_dim, bias=True),
            'time': nn.Linear(self.time_dim, self.node_dim, bias=True),
            'struct': nn.Linear(1, self.node_dim, bias=True),
        })
        self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)

        # --- Model Components ---
        self.local_student_updater = DyKT_Seq(edge_dim=self.node_dim, node_dim=self.node_dim)
        self.question_updater = DyKT_Seq(edge_dim=self.node_dim, node_dim=self.node_dim)
        self.gnn_fusion = GCNConv(self.node_dim, self.node_dim)
        self.feature_fusion_norm = nn.LayerNorm(self.node_dim)

        self.final_fusion_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2, self.node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_layer = nn.Linear(self.node_dim, self.node_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def get_dygkt_features(self, neighbor_node_ids, neighbor_edge_ids, neighbor_times,
                             current_dst_node_ids, current_dst_skill_ids):
        """Computes a full feature set for a batch of sequences, DyGKT-style."""
        node_feat = self.projection_layer['feature_linear'](self.node_raw_features[neighbor_node_ids])
        edge_feat = self.projection_layer['edge'](self.edge_raw_features[neighbor_edge_ids][..., 0].unsqueeze(-1))
        time_feat = self.projection_layer['time'](self.time_encoder(neighbor_times))
        co_occurrence_feat = (neighbor_node_ids == current_dst_node_ids.unsqueeze(1)).float().unsqueeze(-1)
        struct_emb = self.projection_layer['struct'](co_occurrence_feat)
        skill_ids_in_history = self.node_raw_features[neighbor_node_ids][..., 0].long()
        skill_similarity_feat = (skill_ids_in_history == current_dst_skill_ids.unsqueeze(1)).float().unsqueeze(-1)
        skill_emb = self.projection_layer['struct'](skill_similarity_feat)
        combined_features = node_feat + edge_feat + time_feat + struct_emb + skill_emb
        return self.feature_fusion_norm(combined_features)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray, # Added for API consistency
                                                 retrieved_indices: np.ndarray): # The crucial new input
        batch_size = len(src_node_ids)
        src_node_ids_t = torch.from_numpy(src_node_ids).long().to(self.device)
        dst_node_ids_t = torch.from_numpy(dst_node_ids).long().to(self.device)
        current_skill_ids = self.node_raw_features[dst_node_ids_t][:, 0].long()

        # === Part 1: LOCAL CONTEXT (Original DyGKT Path) ===
        local_neighbor_ids, local_edge_ids, local_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors
        )
        local_neighbor_ids_t = torch.from_numpy(local_neighbor_ids).long().to(self.device)
        local_edge_ids_t = torch.from_numpy(local_edge_ids).long().to(self.device)
        local_times_t = torch.from_numpy(local_times).float().to(self.device)
        
        local_history_features = self.get_dygkt_features(
            local_neighbor_ids_t, local_edge_ids_t, local_times_t, dst_node_ids_t, current_skill_ids
        )
        local_student_emb = self.local_student_updater.update(local_history_features)

        # === Part 2: GLOBAL CONTEXT (RAG Path) ===
        retrieved_indices_t = torch.from_numpy(retrieved_indices).long().to(self.device)
        
        graph_data_list = []
        for i in range(batch_size):
            demo_nodes_in_batch = self.retrieval_pool_nodes[retrieved_indices_t[i]].flatten()
            valid_nodes = demo_nodes_in_batch[demo_nodes_in_batch > 0].unique()

            if len(valid_nodes) > 0:
                node_map = {orig_id.item(): new_id for new_id, orig_id in enumerate(valid_nodes)}
                edge_list = []
                for demo_idx in retrieved_indices_t[i]:
                    demo_node_seq = self.retrieval_pool_nodes[demo_idx]
                    valid_demo_nodes = demo_node_seq[demo_node_seq > 0]
                    if len(valid_demo_nodes) > 1:
                        for j in range(len(valid_demo_nodes) - 1):
                            u, v = valid_demo_nodes[j].item(), valid_demo_nodes[j+1].item()
                            if u in node_map and v in node_map:
                                edge_list.append([node_map[u], node_map[v]])
                if edge_list:
                    edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                graph_x = self.projection_layer['feature_linear'](self.node_raw_features[valid_nodes])
                graph_data_list.append(Data(x=graph_x, edge_index=edge_index))
            else:
                graph_data_list.append(Data(x=torch.zeros((1, self.node_dim), device=self.device),
                                            edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device)))

        batched_graph = Batch.from_data_list(graph_data_list).to(self.device)
        gnn_output = torch.relu(self.gnn_fusion(batched_graph.x, batched_graph.edge_index))
        global_context_emb = global_mean_pool(gnn_output, batched_graph.batch)

        # === Part 3: FINAL FUSION & PREDICTION ===
        combined_student_emb = torch.cat([local_student_emb, global_context_emb], dim=1)
        final_student_emb = self.final_fusion_mlp(combined_student_emb)

        # === Question Embedding (as in original DyGKT) ===
        dst_neighbor_ids, dst_edge_ids, dst_times = self.neighbor_sampler.get_historical_neighbors(
            dst_node_ids, node_interact_times, self.num_neighbors
        )
        dst_history_features = self.get_dygkt_features(
            torch.from_numpy(dst_neighbor_ids).long().to(self.device),
            torch.from_numpy(dst_edge_ids).long().to(self.device),
            torch.from_numpy(dst_times).float().to(self.device),
            src_node_ids_t, self.node_raw_features[src_node_ids_t][:, 0].long()
        )
        question_emb = self.question_updater.update(dst_history_features)

        final_student_emb = self.output_layer(final_student_emb)
        final_question_emb = self.output_layer(question_emb)

        return self.dropout_layer(final_student_emb), self.dropout_layer(final_question_emb)

print("New RAG4DyG model class defined.")
