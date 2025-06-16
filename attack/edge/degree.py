from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score  
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dhg import Hypergraph
from pure_eval import Evaluator

from parameter_parser import parameter_parser  
from attack_method import get_posteriors, evaluate_attack_performance  

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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

def forget_edges(edge_list, forget_ratio):
    num_edges = len(edge_list)
    num_forget = int(num_edges * forget_ratio)
    forget_indices = torch.randperm(num_edges)[:num_forget]
    new_edge_list = [edge_list[i] for i in range(num_edges) if i not in forget_indices]
    return new_edge_list

def select_top_hyperedges_by_degree(edge_list, top_ratio=0.2):
    
    edge_scores = [(len(edge), edge) for edge in edge_list]

   
    num_edges_to_select = int(len(edge_list) * top_ratio)
    edge_scores.sort(reverse=True, key=lambda x: x[0])
    selected_edges = [edge for _, edge in edge_scores[:num_edges_to_select]]

    return selected_edges

def train(net, X, G, lbl, train_mask, optimizer, epoch, device):
    net.train()
    start_time = time.time()
    optimizer.zero_grad()

    # Ensure tensors are on the correct device
    X, G, lbl, train_mask = X.to(device), G.to(device), lbl.to(device), train_mask.to(device)

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
        X, G, lbl, mask = X.to(device), G.to(device), lbl.to(device), mask.to(device)
        out = net(X, G)
        loss = F.cross_entropy(out[mask], lbl[mask])
        pred = out[mask].max(1)[1]
        correct = pred.eq(lbl[mask]).sum().item()
        acc = correct / mask.sum().item()
    if test:
        print(f"Test set results: loss= {loss.item():.5f}, accuracy= {acc:.5f}")
    return loss.item(), acc

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = torch.eye(data["num_vertices"]).to(device), data["labels"].to(device)
    #X = data["features"].to(device)
    #lbl = data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"])

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    return X, lbl, G, net, optimizer

def train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, num_epochs, device):
    best_state = None
    best_epoch, best_val = 0, 0

    start_time = time.time()

    for epoch in range(num_epochs):
        train(net, X, G, lbl, train_mask, optimizer, epoch, device)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask, device)[1]
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())

    end_time = time.time()
    total_training_time = end_time - start_time

    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    print(f"total training time: {total_training_time:.2f} seconds")

    return best_state, best_epoch, total_training_time

def test_model(net, best_state, X, G, lbl, test_mask, device):
    print("test...")
    net.load_state_dict(best_state)
    loss, acc = infer(net, X, G, lbl, test_mask, device, test=True)

    return loss, acc

def print_statistics(train_mask, val_mask, test_mask):
    num_train = torch.sum(train_mask).item()
    num_val = torch.sum(val_mask).item()
    num_test = torch.sum(test_mask).item()

    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")

def evaluate_attack_performance(original_posteriors, forgetting_posteriors, train_mask, test_mask):
    # Move all relevant tensors to the same device
    original_posteriors = original_posteriors.to(train_mask.device)
    forgetting_posteriors = forgetting_posteriors.to(train_mask.device)
    train_mask = train_mask.to(train_mask.device)
    test_mask = test_mask.to(test_mask.device)

    # Compute distances or other metrics
    train_distances = (original_posteriors[train_mask] - forgetting_posteriors[train_mask]).norm(p=2, dim=1)
    test_distances = (original_posteriors[test_mask] - forgetting_posteriors[test_mask]).norm(p=2, dim=1)

    # AUC calculation
    labels = torch.cat([torch.ones(train_mask.sum().item()), torch.zeros(test_mask.sum().item())])
    scores = torch.cat([train_distances, test_distances])
    auc_value = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())

    print("Train distances:", train_distances.mean().item())
    print("Test distances:", test_distances.mean().item())
    print(f"Attack AUC: {auc_value:.5f}")  # Print AUC value

def main():
    args = parameter_parser()

    if args['random_seed'] is not None:
        random_seed = args['random_seed']
    else:
        random_seed = int(time.time())
    set_seed(random_seed)
    device = torch.device(args['device']) if args['device'] == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data_class = args['dataset_class']
    data = data_class()

    total_vertices = data["num_vertices"]
    train_mask, val_mask, test_mask = split_data(total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio'])

    X, lbl, G, net, optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'], args['weight_decay'])

    # unlearning edge
    new_edge_list = forget_edges(data['edge_list'], args['forget_ratio'])

    selected_edges = select_top_hyperedges_by_degree(new_edge_list, top_ratio=0.4)

    G = Hypergraph(total_vertices, selected_edges).to(device)  # Ensure G is on the correct device

    print("Training forgetting model...")
    best_state, best_epoch, total_training_time = train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, args['num_epochs'], device)
    loss, acc = test_model(net, best_state, X, G, lbl, test_mask, device)
    print(f"Forgetting model final result: epoch: {best_epoch}")
    print(f"Forgetting model test loss: {loss:.5f}, test accuracy: {acc:.5f}")
    print(f"Forgetting model training time: {total_training_time:.2f} seconds")

    print(f"Number of hyperedges used for training: {len(selected_edges)}")

    with open('../original_posteriors.pkl', 'rb') as f:
        original_posteriors = pickle.load(f)

    original_posteriors = torch.tensor(original_posteriors).to(device)

    forgetting_posteriors = get_posteriors(net, X, G).cpu()

    evaluate_attack_performance(original_posteriors, forgetting_posteriors, train_mask, test_mask)

    print_statistics(train_mask, val_mask, test_mask)

if __name__ == "__main__":
    main()
