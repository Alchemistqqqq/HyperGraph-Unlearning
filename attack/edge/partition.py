from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from dhg import Hypergraph
from pure_eval import Evaluator
from parameter_parser import parameter_parser  
from aggregation_methods import majority_voting, average_voting, weighted_average_voting 
from attack_method import get_posteriors, evaluate_attack_performance  

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

def forget_edges(edge_list, forget_ratio):
    num_edges = len(edge_list)
    num_forget = int(num_edges * forget_ratio)
    forget_indices = torch.randperm(num_edges)[:num_forget]
    new_edge_list = [edge_list[i] for i in range(num_edges) if i not in forget_indices]
    
    forgotten_nodes = set()
    for i in forget_indices.tolist():
        forgotten_nodes.update(edge_list[i])
    forgotten_nodes = torch.tensor(sorted(list(forgotten_nodes)))

    print(f"Total edges: {num_edges}")
    print(f"Edges to forget: {len(forget_indices)}")
    print(f"Remaining edges: {len(new_edge_list)}")
    return new_edge_list, forgotten_nodes

def train(net, X, G, lbl, train_mask, optimizer, epoch):
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

def infer(net, X, G, lbl, mask, test=False):
    net.eval()
    with torch.no_grad():
        out = net(X, G)
        loss = F.cross_entropy(out[mask], lbl[mask])
        pred = out[mask].max(1)[1]
        correct = pred.eq(lbl[mask]).sum().item()
        acc = correct / mask.sum().item()
    if test:
        print(f"Test set results: loss= {loss.item():.5f}, accuracy= {acc:.5f}")
    return acc, pred

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = torch.eye(data["num_vertices"]).to(device), data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"]).to(device)

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return X, lbl, G, net, optimizer

def train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, num_epochs, device):
    best_state = None
    best_epoch, best_val = 0, 0

    start_time = time.time()

    for epoch in range(num_epochs):
        train(net, X, G, lbl, train_mask.to(device), optimizer, epoch)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res, _ = infer(net, X, G, lbl, val_mask.to(device))
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

    return best_state, best_epoch

def test_model(net, best_state, X, G, lbl, test_mask, device):
    print("test...")
    net.load_state_dict(best_state)
    res, pred = infer(net, X, G, lbl, test_mask.to(device), test=True)

    return res, pred

def print_statistics(train_mask, val_mask, test_mask):
    num_train = torch.sum(train_mask).item()
    num_val = torch.sum(val_mask).item()
    num_test = torch.sum(test_mask).item()

    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")

def split_into_subgraphs(edge_list, num_subgraphs, num_vertices):
    num_edges = len(edge_list)
    subgraph_size = num_edges // num_subgraphs

    random.shuffle(edge_list)  

    subgraphs = []
    subgraph_edge_counts = []
    for i in range(num_subgraphs):
        start_idx = i * subgraph_size
        end_idx = (i + 1) * subgraph_size if i != num_subgraphs - 1 else num_edges
        sub_edge_list = edge_list[start_idx:end_idx]
        subgraphs.append(Hypergraph(num_vertices, sub_edge_list))
        subgraph_edge_counts.append(len(sub_edge_list))

    return subgraphs, subgraph_edge_counts

def get_posteriors(net, X, G):
    net.eval()
    with torch.no_grad():
        out = net(X, G)
    return out

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

    X, lbl, G, net, optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'], args['weight_decay'])

    new_edge_list, forget_indices = forget_edges(data['edge_list'], args['forget_ratio'])
    G = Hypergraph(total_vertices, new_edge_list).to(device)

    subgraphs, subgraph_edge_counts = split_into_subgraphs(new_edge_list, args['num_subgraphs'], total_vertices)

    all_predictions = []

    original_posteriors = get_posteriors(net, X, G).to(device)

    for i, subgraph in enumerate(subgraphs):
        print(f"Training on subgraph {i + 1} with {subgraph_edge_counts[i]} edges")
        sub_net, sub_optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'], args['weight_decay'])[3:5]
        best_state, best_epoch = train_and_evaluate_model(sub_net, X, subgraph.to(device), lbl, train_mask, val_mask, sub_optimizer, args['num_epochs'], device)

        print(f"Testing on subgraph {i + 1}")
        res, pred = test_model(sub_net, best_state, X, subgraph.to(device), lbl, test_mask, device)
        print(f"Subgraph {i + 1} result: {res}")
        print(f"Subgraph {i + 1} edge count: {subgraph_edge_counts[i]}")  

        all_predictions.append(pred)

    if args['aggregation_method'] == 'majority_voting':
        final_predictions = majority_voting(all_predictions)
    elif args['aggregation_method'] == 'average_voting':
        final_predictions = average_voting(all_predictions)
    elif args['aggregation_method'] == 'weighted_average_voting':
        weights = torch.ones(len(all_predictions)) / len(all_predictions)
        final_predictions = weighted_average_voting(all_predictions, weights)
    else:
        raise ValueError(f"Unknown aggregation method: {args['aggregation_method']}")

    correct = final_predictions.eq(lbl[test_mask]).sum().item()
    final_accuracy = correct / test_mask.sum().item()

    print(f"Final Test Accuracy after {args['aggregation_method']}: {final_accuracy:.5f}")

    print_statistics(train_mask, val_mask, test_mask)

    current_posteriors = get_posteriors(net, X, G)

    evaluate_attack_performance(original_posteriors, current_posteriors, train_mask, forget_indices, test_mask)


if __name__ == "__main__":
    main()
