import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import networkx as nx
from torch_geometric.utils import from_networkx

from utils.utils import NeighborSampler
from models.modules import TimeDualDecayEncoder

# Assume the original DyKT_Seq (GRU updater) is available from your code
class DyKT_Seq(nn.Module):
    def __init__(self, edge_dim : int, node_dim: int):
        super(DyKT_Seq,self).__init__()
        self.hid_node_updater = nn.GRU(input_size=edge_dim, hidden_size=node_dim, batch_first=True)
    def update(self, x):
        _, hidden = self.hid_node_updater(x)
        return torch.squeeze(hidden, dim=0)

class RAG_DyGKT(nn.Module):
    """
    A true Retrieval-Augmented Generation model for Knowledge Tracing,
    combining the strengths of DyGKT and the RAG4DyG framework.

    Workflow:
    1. Local Context Path (DyGKT):
       - Retrieve the student's own recent history.
       - Create DyGKT features and process with a GRU to get a `local_emb`.
    2. Global Context Path (RAG):
       - Use pre-computed retrieved demonstration indices (`retrieved_indices`).
       - Build a single summary graph from these k demonstration sequences.
       - Process the summary graph with a GNN to get a `global_emb`.
    3. Fusion & Prediction:
       - Concatenate local_emb and global_emb.
       - Project with an MLP to get the final student embedding.
       - Predict the link score against the question embedding.
    """
    def __init__(self,
                 node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 retrieval_pool_features: dict, # A dict holding features of all sequences in the retrieval pool
                 time_dim: int = 16,
                 num_neighbors: int = 50, # For local history
                 dropout: float = 0.2,
                 device: str = 'cuda:0',
                 **kwargs):

        super(RAG_DyGKT, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device

        # --- Feature Storage ---
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        # The retrieval pool contains all historical sequences from the training set.
        # We need their features readily available.
        self.retrieval_pool_nodes = torch.from_numpy(retrieval_pool_features['nodes']).long().to(device)
        self.retrieval_pool_edges = torch.from_numpy(retrieval_pool_features['edges']).long().to(device)
        self.retrieval_pool_times = torch.from_numpy(retrieval_pool_features['times']).float().to(device)

        self.node_dim = 64
        self.time_dim = time_dim

        # --- DyGKT Projection & Encoders ---
        self.projection_layer = nn.ModuleDict({
            'feature_linear': nn.Linear(self.node_raw_features.shape[1], self.node_dim, bias=True),
            'edge': nn.Linear(1, self.node_dim, bias=True),
            'time': nn.Linear(self.time_dim, self.node_dim, bias=True),
            'struct': nn.Linear(1, self.node_dim, bias=True),
        })
        self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)

        # --- Model Components ---
        # 1. Local History Processor (from original DyGKT)
        self.local_student_updater = DyKT_Seq(edge_dim=self.node_dim, node_dim=self.node_dim)
        self.question_updater = DyKT_Seq(edge_dim=self.node_dim, node_dim=self.node_dim) # For question embedding

        # 2. Global Context Fuser (GNN for RAG)
        self.gnn_fusion = GCNConv(self.node_dim, self.node_dim)
        self.feature_fusion_norm = nn.LayerNorm(self.node_dim) # Good idea to keep this!

        # 3. Final Fusion Layer
        self.final_fusion_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2, self.node_dim), # local_emb + global_emb
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
        # Project base features
        node_feat = self.projection_layer['feature_linear'](self.node_raw_features[neighbor_node_ids])
        edge_feat = self.projection_layer['edge'](self.edge_raw_features[neighbor_edge_ids][..., 0].unsqueeze(-1))
        time_feat = self.projection_layer['time'](self.time_encoder(neighbor_times))

        # Structural features
        co_occurrence_feat = (neighbor_node_ids == current_dst_node_ids.unsqueeze(1)).float().unsqueeze(-1)
        struct_emb = self.projection_layer['struct'](co_occurrence_feat)

        skill_ids_in_history = self.node_raw_features[neighbor_node_ids][..., 0].long()
        skill_similarity_feat = (skill_ids_in_history == current_dst_skill_ids.unsqueeze(1)).float().unsqueeze(-1)
        skill_emb = self.projection_layer['struct'](skill_similarity_feat)
        
        # Sum all features and normalize
        combined_features = node_feat + edge_feat + time_feat + struct_emb + skill_emb
        return self.feature_fusion_norm(combined_features)


    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 retrieved_indices: np.ndarray): # Pass retrieved indices here!
        batch_size = len(src_node_ids)
        src_node_ids_t = torch.from_numpy(src_node_ids).long().to(self.device)
        dst_node_ids_t = torch.from_numpy(dst_node_ids).long().to(self.device)
        current_skill_ids = self.node_raw_features[dst_node_ids_t][:, 0].long()

        # === Part 1: LOCAL CONTEXT (Original DyGKT Path) ===
        local_neighbor_ids, local_edge_ids, local_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors
        )
        local_neighbor_ids = torch.from_numpy(local_neighbor_ids).long().to(self.device)
        local_edge_ids = torch.from_numpy(local_edge_ids).long().to(self.device)
        local_times = torch.from_numpy(local_times).float().to(self.device)
        
        # Get features for the local history
        local_history_features = self.get_dygkt_features(
            local_neighbor_ids, local_edge_ids, local_times, dst_node_ids_t, current_skill_ids
        )
        # Process with GRU to get local embedding
        local_student_emb = self.local_student_updater.update(local_history_features)

        # === Part 2: GLOBAL CONTEXT (RAG Path) ===
        retrieved_indices_t = torch.from_numpy(retrieved_indices).long().to(self.device) # (batch_size, k)
        k = retrieved_indices_t.shape[1]

        graph_data_list = []
        for i in range(batch_size):
            # For each student in the batch, build one summary graph from their k retrieved demos
            demo_nodes_list = []
            demo_edges = []
            
            # Use NetworkX for easier graph construction
            summary_graph = nx.Graph()
            
            for demo_idx in retrieved_indices_t[i]:
                # Get the node sequence of the demo
                demo_node_seq = self.retrieval_pool_nodes[demo_idx]
                # Add edges for this sequence (e.g., sequential, or all-to-all, depends on strategy)
                valid_nodes = demo_node_seq[demo_node_seq > 0] # Filter out padding
                if len(valid_nodes) > 1:
                    summary_graph.add_edges_from(zip(valid_nodes[:-1].tolist(), valid_nodes[1:].tolist()))
            
            if len(summary_graph.nodes) > 0:
                # Convert to PyG Data object
                pyg_graph = from_networkx(summary_graph)
                # Assign features to the nodes in the graph
                node_ids_in_graph = torch.tensor(list(summary_graph.nodes), dtype=torch.long, device=self.device)
                pyg_graph.x = self.projection_layer['feature_linear'](self.node_raw_features[node_ids_in_graph])
                graph_data_list.append(pyg_graph)
            else:
                 # Handle empty retrieval case
                graph_data_list.append(Data(x=torch.zeros((1, self.node_dim), device=self.device),
                                            edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device)))

        # Batch all summary graphs together
        batched_graph = Batch.from_data_list(graph_data_list).to(self.device)

        # Process the batch of graphs with GNN and pool to get one embedding per graph
        gnn_output = torch.relu(self.gnn_fusion(batched_graph.x, batched_graph.edge_index))
        global_context_emb = global_mean_pool(gnn_output, batched_graph.batch)

        # === Part 3: FINAL FUSION & PREDICTION ===
        # Concatenate local and global context embeddings
        combined_student_emb = torch.cat([local_student_emb, global_context_emb], dim=1)
        final_student_emb = self.final_fusion_mlp(combined_student_emb)

        # === Question Embedding (as in original DyGKT) ===
        dst_neighbor_ids, dst_edge_ids, dst_times = self.neighbor_sampler.get_historical_neighbors(
            dst_node_ids, node_interact_times, self.num_neighbors
        )
        dst_neighbor_ids = torch.from_numpy(dst_neighbor_ids).long().to(self.device)
        dst_edge_ids = torch.from_numpy(dst_edge_ids).long().to(self.device)
        dst_times = torch.from_numpy(dst_times).float().to(self.device)
        
        # Here, the "destination" for the question's history is the current student
        question_history_features = self.get_dygkt_features(
            dst_neighbor_ids, dst_edge_ids, dst_times, src_node_ids_t, self.node_raw_features[src_node_ids_t][:, 0].long()
        )
        question_emb = self.question_updater.update(question_history_features)

        # Apply final output layer
        final_student_emb = self.output_layer(final_student_emb)
        final_question_emb = self.output_layer(question_emb)

        return self.dropout_layer(final_student_emb), self.dropout_layer(final_question_emb)
