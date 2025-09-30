import torch
import torch.nn.functional as F

def cross_modality_label_preserving_loss(f, f_mapped, labels, tau=0.6):

    B = f.size(0)
    sim_matrix = F.normalize(f, dim=1).mm(F.normalize(f_mapped, dim=1).t())

    same_cluster_mask = labels.view(-1, 1) == labels.view(1, -1)

    sim_pos = sim_matrix.masked_fill(~same_cluster_mask, float('inf'))
    hardest_pos_sim = torch.min(sim_pos, dim=1).values

    unique_labels = torch.unique(labels)
    C = unique_labels.size(0)

    max_sim_per_cluster = torch.full((B, C), -float('inf'), device=f.device)

    for c_idx, c_label in enumerate(unique_labels):
        cluster_mask = (labels == c_label)
        if cluster_mask.any():
            cluster_sim = sim_matrix[:, cluster_mask]
            max_sim_per_cluster[:, c_idx] = torch.max(cluster_sim, dim=1).values

    self_cluster_mask = (unique_labels.view(1, -1) == labels.view(-1, 1))
    max_sim_per_cluster[self_cluster_mask] = -float('inf')

    numerator = torch.exp(hardest_pos_sim / tau)

    denominator = torch.sum(torch.exp(max_sim_per_cluster / tau), dim=1)

    valid_mask = (denominator > 0)
    if not valid_mask.any():
        return torch.tensor(0.0, device=f.device)

    loss = -torch.log(numerator / denominator).mean()

    return loss