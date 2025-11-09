import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# NOTE: This is the original, un-edited version of the file from the repository.

def evaluate_model_link_classification(model_name: str, model: nn.Module, neighbor_sampler,
                                       evaluate_idx_data_loader: torch.utils.data.DataLoader,
                                       evaluate_neg_edge_sampler, evaluate_data: object,
                                       loss_func: nn.Module, num_neighbors: int = 20):
    """
    evaluate models on the link classification task
    """
    # Ensures the code still works for models that don't have retrieve_latest_mailboxes
    if model_name in ['TGN', 'CAWN', 'TCL']:
        # Put the model in evaluation mode
        model.eval()
        # memory-based models need to retrieve messages for evaluation
        model[0].retrieve_latest_mailboxes()
    elif model_name in ['Jodie', 'DyRep', 'DyGFormer']:
        # Put the model in evaluation mode
        model.eval()
        # memory-based models need to retrieve memory for evaluation
        model[0].retrieve_memory()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            pos_src_node_ids, pos_dst_node_ids, pos_node_interact_times, pos_edge_ids = \
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids
            neg_src_node_ids, neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(pos_src_node_ids))

            if model_name in ['TGN', 'CAWN', 'TCL']:
                pos_src_node_embeddings, pos_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(pos_src_node_ids, pos_dst_node_ids, pos_node_interact_times,
                                                                      pos_edge_ids, True, num_neighbors)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(neg_src_node_ids, neg_dst_node_ids, pos_node_interact_times,
                                                                      None, False, num_neighbors)
            elif model_name in ['Jodie', 'DyRep', 'TGAT', 'DyGFormer']:
                pos_src_node_embeddings, pos_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(pos_src_node_ids, pos_dst_node_ids, pos_node_interact_times,
                                                                      pos_edge_ids, num_neighbors)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(neg_src_node_ids, neg_dst_node_ids, pos_node_interact_times,
                                                                      None, num_neighbors)
            elif model_name in ['RAG4DyG', 'DyGKT', 'DKT', 'CTNCM', 'AKT', 'simpleKT', 'QIKT', 'IEKT', 'IPKT', 'DIMKT']:
                pos_src_node_embeddings, pos_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(pos_src_node_ids, pos_edge_ids, pos_node_interact_times, pos_dst_node_ids)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(neg_src_node_ids, None, pos_node_interact_times, neg_dst_node_ids)
            else:
                raise ValueError(f"Wrong model name {model_name}!")

            pos_probabilities = model[1](pos_src_node_embeddings, pos_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            neg_probabilities = model[1](neg_src_node_embeddings, neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([pos_probabilities, neg_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(pos_probabilities), torch.zeros_like(neg_probabilities)], dim=0)

            loss = loss_func(predicts, labels)
            evaluate_losses.append(loss.item())

            evaluate_metrics.append({
                'average_precision': average_precision_score(y_true=labels.cpu().numpy(), y_score=predicts.cpu().numpy()),
                'roc_auc': roc_auc_score(y_true=labels.cpu().numpy(), y_score=predicts.cpu().numpy())
            })
            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    mean_evaluate_loss = np.mean(evaluate_losses)
    mean_evaluate_metrics = {
        'average_precision': np.mean([single_metric['average_precision'] for single_metric in evaluate_metrics]),
        'roc_auc': np.mean([single_metric['roc_auc'] for single_metric in evaluate_metrics])
    }
    return mean_evaluate_loss, mean_evaluate_metrics
