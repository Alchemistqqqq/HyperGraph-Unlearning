import torch
from sklearn.metrics import roc_auc_score

def get_posteriors(net, X, G):
    net.eval()
    with torch.no_grad():
        out = net(X, G)
    return out

def evaluate_attack_performance(original_posteriors, unlearned_posteriors, train_mask, test_mask, distance_metric='l2_norm'):
    original_train_posteriors = original_posteriors[train_mask]
    unlearned_train_posteriors = unlearned_posteriors[train_mask]

    original_test_posteriors = original_posteriors[test_mask]
    unlearned_test_posteriors = unlearned_posteriors[test_mask]

    if distance_metric == 'l2_norm':
        train_distances = (original_train_posteriors - unlearned_train_posteriors).norm(p=2, dim=1)
        test_distances = (original_test_posteriors - unlearned_test_posteriors).norm(p=2, dim=1)
    elif distance_metric == 'manhattan':
        train_distances = (original_train_posteriors - unlearned_train_posteriors).abs().sum(dim=1)
        test_distances = (original_test_posteriors - unlearned_test_posteriors).abs().sum(dim=1)
    elif distance_metric == 'chebyshev':
        train_distances = (original_train_posteriors - unlearned_train_posteriors).abs().max(dim=1)[0]
        test_distances = (original_test_posteriors - unlearned_test_posteriors).abs().max(dim=1)[0]
    elif distance_metric == 'cosine':
        train_distances = 1 - torch.nn.functional.cosine_similarity(original_train_posteriors, unlearned_train_posteriors)
        test_distances = 1 - torch.nn.functional.cosine_similarity(original_test_posteriors, unlearned_test_posteriors)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    # Labels: 1 for training samples (member), 0 for test samples (non-member)
    labels = torch.cat((torch.ones(train_distances.size(0)), torch.zeros(test_distances.size(0))))
    distances = torch.cat((train_distances, test_distances))

    auc = roc_auc_score(labels.cpu().numpy(), distances.cpu().numpy())
    print(f"Attack AUC: {auc:.5f}")
