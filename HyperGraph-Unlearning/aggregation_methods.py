import torch
import torch.nn.functional as F

def majority_voting(predictions):
    stacked_predictions = torch.stack(predictions, dim=1)
    majority_vote, _ = torch.mode(stacked_predictions, dim=1)
    return majority_vote

def average_voting(predictions):
    stacked_predictions = torch.stack(predictions, dim=1)
    average_vote = stacked_predictions.float().mean(dim=1)
    return torch.round(average_vote).long()

def weighted_average_voting(predictions, weights):
    stacked_predictions = torch.stack(predictions, dim=1)
    weighted_sum = torch.sum(stacked_predictions.float() * weights.view(-1, 1), dim=1)
    return torch.round(weighted_sum).long()


