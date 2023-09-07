import torch
import torch.nn.functional as F


def get_similarity(pseudo_label, targets_u_s, dim=1):
    log_w = torch.log(pseudo_label)
    log_s = torch.log(targets_u_s)
    coefficient = torch.sum(torch.exp((log_w + log_s) / 2), dim=dim)
    return coefficient


def get_distance_loss(pseudo_label, logits_u_s, dim=1):
    targets_u_s = torch.softmax(logits_u_s.detach(), dim=1)
    return 1. - get_similarity(pseudo_label, targets_u_s, dim).mean()
