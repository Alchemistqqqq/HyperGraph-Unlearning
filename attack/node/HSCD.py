from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from dhg import Hypergraph
from pure_eval import Evaluator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from parameter_parser import parameter_parser 
from attack_method import evaluate_attack_performance
from attack_method import get_posteriors

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def split_data(total_vertices, train_ratio, val_ratio, test_ratio):
    all_indices = torch.randperm(total_vertices)

    train_size = int(total_vertices * train_ratio)
    val_size = int(total_vertices * val_ratio)
    test_size = total_vertices - train_size - val_size

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    train_mask = torch.zeros(total_vertices, dtype=torch.bool)
    val_mask = torch.zeros(total_vertices, dtype=torch.bool)
    test_mask = torch.zeros(total_vertices, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

def forget_nodes(total_vertices, train_mask, forget_ratio):
    train_indices = torch.nonzero(train_mask).squeeze()
    num_forget = int(len(train_indices) * forget_ratio)
    forget_indices = train_indices[torch.randperm(len(train_indices))[:num_forget]].tolist()

    new_train_mask = train_mask.clone()
    new_train_mask[forget_indices] = False

    print(f"Total nodes: {total_vertices}")
    print(f"Nodes to forget: {len(forget_indices)}")
    print(f"Remaining training nodes: {new_train_mask.sum().item()}")

    return new_train_mask, forget_indices

def train(net, X, G, lbl, train_mask, optimizer, epoch, device):
    net.train()
    start_time = time.time()
    optimizer.zero_grad()
    out = net(X, G)
    loss = F.cross_entropy(out[train_mask], lbl[train_mask])
    loss.backward()
    optimizer.step()
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch: {epoch}, Loss: {loss.item():.5f}, Time: {epoch_time:.5f}s")

def infer(net, X, G, lbl, mask, device, test=False):
    net.eval()
    with torch.no_grad():
        out = net(X, G)
        loss = F.cross_entropy(out[mask], lbl[mask])
        pred = out[mask].max(1)[1]
        correct = pred.eq(lbl[mask]).sum().item()
        acc = correct / mask.sum().item()
    if test:
        print(f"Test set results: loss= {loss.item():.5f}, accuracy= {acc:.5f}")
    return acc, out

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = torch.eye(data["num_vertices"]).to(device), data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"]).to(device)

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return X, lbl, G, net, optimizer

def print_statistics(train_mask, val_mask, test_mask, forget_indices=None):
    num_train = torch.sum(train_mask).item()
    num_val = torch.sum(val_mask).item()
    num_test = torch.sum(test_mask).item()

    print(f"\n=== Dataset Statistics ===")
    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")
    
    if forget_indices is not None:
        print(f"\n=== Forgetting Statistics ===")
        print(f"Number of forgotten nodes: {len(forget_indices)}")

def main():
    args = parameter_parser()

    if args['random_seed'] is not None:
        random_seed = args['random_seed']
    else:
        random_seed = int(time.time())
    set_seed(random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data = args['dataset_class']()

    total_vertices = data['num_vertices']
    train_mask, val_mask, test_mask = split_data(total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio'])

    # Initial full model training
    print("=== Training Initial Model ===")
    X, lbl, G, net, optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'], args['weight_decay'])
    
    best_state = None
    best_val = 0
    for epoch in range(args['num_epochs']):
        train(net, X, G, lbl, train_mask, optimizer, epoch, device)
        val_acc, _ = infer(net, X, G, lbl, val_mask, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = deepcopy(net.state_dict())
    
    net.load_state_dict(best_state)
    test_acc, original_out = infer(net, X, G, lbl, test_mask, device, test=True)
    original_posteriors = get_posteriors(net, X, G)
    #with open('../original_posteriors.pkl', 'rb') as f:
        #original_posteriors = torch.tensor(pickle.load(f)).to(device)
    
    # Node forgetting
    print("\n=== Applying Node Forgetting ===")
    new_train_mask, forget_indices = forget_nodes(total_vertices, train_mask, args['forget_ratio'])
    
    # Retrain with forgotten nodes removed
    print("\n=== Retraining After Forgetting ===")
    net = args['model_class'](X.shape[1], 32, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    
    best_state = None
    best_val = 0
    for epoch in range(args['num_epochs']):
        train(net, X, G, lbl, new_train_mask, optimizer, epoch, device)
        val_acc, _ = infer(net, X, G, lbl, val_mask, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = deepcopy(net.state_dict())
    
    net.load_state_dict(best_state)
    unlearned_test_acc, unlearned_out = infer(net, X, G, lbl, test_mask, device, test=True)
    unlearned_posteriors = get_posteriors(net, X, G)
    
    # Evaluation
    print("\n=== Final Evaluation ===")
    evaluate_attack_performance(
        original_posteriors,
        unlearned_posteriors,
        train_mask.to(device),
        forget_indices,
        test_mask.to(device)
    )
    
    print(f"\nOriginal Test Accuracy: {test_acc:.4f}")
    print(f"Unlearned Test Accuracy: {unlearned_test_acc:.4f}")
    print_statistics(new_train_mask, val_mask, test_mask, forget_indices)

if __name__ == "__main__":
    main()