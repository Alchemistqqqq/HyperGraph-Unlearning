import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def get_posteriors(net, X, G):
    net.eval()
    with torch.no_grad():
        out = net(X, G)
    return out

def evaluate_attack_performance(original_posteriors, unlearned_posteriors, 
                              train_mask, forget_indices, test_mask):
    # Convert to numpy for sklearn
    original_posteriors = original_posteriors.cpu().numpy()
    unlearned_posteriors = unlearned_posteriors.cpu().numpy()
    train_mask = train_mask.cpu().numpy()
    test_mask = test_mask.cpu().numpy()
    
    # Get indices
    forget_indices = np.array(forget_indices)
    test_indices = np.where(test_mask)[0]
    
    # Sample equal number of points from each group
    min_size = min(len(forget_indices), len(test_indices), 5000)
    forget_sample = np.random.choice(forget_indices, min_size, replace=False)
    test_sample = np.random.choice(test_indices, min_size, replace=False)
    
    # Calculate distances (L2 norm between original and unlearned posteriors)
    forget_distances = np.linalg.norm(
        original_posteriors[forget_sample] - unlearned_posteriors[forget_sample], 
        axis=1
    )
    test_distances = np.linalg.norm(
        original_posteriors[test_sample] - unlearned_posteriors[test_sample],
        axis=1
    )
    
    # Calculate AUC for forgotten vs test nodes
    distances = np.concatenate([forget_distances, test_distances])
    labels = np.concatenate([np.ones(min_size), np.zeros(min_size)])
    auc_score = roc_auc_score(labels, distances)
    
    print(f"\nMembership Inference Attack AUC: {auc_score:.4f}")
    return auc_score