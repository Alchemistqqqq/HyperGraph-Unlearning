import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from copy import deepcopy

from dhg import Hypergraph
from pure_eval import Evaluator

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parameter_parser import parameter_parser  
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

def calculate_degrees(num_vertices, edge_list):
    degrees = {}
    for edge in edge_list:
        for node in edge:
            if node not in degrees:
                degrees[node] = 0
            degrees[node] += 1
    return degrees

def calculate_edge_weights(edge_list, degrees):
    weights = []
    for edge in edge_list:
        min_degree = min(degrees[node] for node in edge)
        weights.append(min_degree)
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights

def normalize_weights(weights):
    return weights / weights.sum()

def calculate_sampling_probabilities(weights, alpha):
    # Adjust alpha to reverse the effect for alpha in [0, 1]
    adjusted_weights = (weights.max() - weights) + weights.min()
    probabilities = (1 - alpha) * adjusted_weights + alpha * weights
    probabilities = normalize_weights(probabilities)
    return probabilities

def forget_edges(edge_list, forget_ratio):
    num_edges = len(edge_list)
    num_forget = int(num_edges * forget_ratio)
    forget_indices = torch.randperm(num_edges)[:num_forget]
    new_edge_list = [edge_list[i] for i in range(num_edges) if i not in forget_indices]
    return new_edge_list

def sample_edges(data, alpha, sample_ratio):
    edge_list = data['edge_list']
    num_vertices = data["num_vertices"]

    # Calculate node degrees
    degrees = calculate_degrees(num_vertices, edge_list)

    # Step 1: Calculate edge weights
    edge_weights = calculate_edge_weights(edge_list, degrees)

    # Step 2: Normalize weights
    normalized_weights = normalize_weights(edge_weights)

    # Step 3: Initialize sets for sampled edges and nodes
    sampled_edges = []
    sampled_nodes = set()

    # Step 4: Calculate the number of edges to sample
    num_edges = len(edge_list)
    num_sampled_edges = int(num_edges * sample_ratio)

    # Step 5: Create a list of remaining edge indices
    remaining_edges = list(range(num_edges))

    # Step 6: Iteratively sample edges without replacement
    while len(sampled_edges) < num_sampled_edges:
        # Calculate sampling probabilities based on current remaining edges
        current_weights = normalized_weights[remaining_edges]
        sampling_probabilities = calculate_sampling_probabilities(current_weights, alpha)

        # Sample one edge from remaining edges
        sampled_index = torch.multinomial(sampling_probabilities, 1, replacement=False).item()
        sampled_edge_idx = remaining_edges[sampled_index]

        # Add the sampled edge to the list of sampled edges
        sampled_edge = edge_list[sampled_edge_idx]
        sampled_edges.append(sampled_edge)

        # Add the nodes of the sampled edge to the set of sampled nodes
        sampled_nodes.update(sampled_edge)

        # Remove the sampled edge from the remaining edges list to prevent resampling
        remaining_edges.pop(sampled_index)

    return sampled_edges, list(sampled_nodes)

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    return X, lbl, G, net, optimizer

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
    return loss.item(), acc

def train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, num_epochs):
    best_state = None
    best_epoch, best_val = 0, 0

    start_time = time.time()

    for epoch in range(num_epochs):
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask)[1]
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

def test_model(net, best_state, X, G, lbl, test_mask):
    print("test...")
    net.load_state_dict(best_state)
    loss, acc = infer(net, X, G, lbl, test_mask, test=True)

    return loss, acc

def print_statistics(train_mask, val_mask, test_mask):
    num_train = torch.sum(train_mask).item()
    num_val = torch.sum(val_mask).item()
    num_test = torch.sum(test_mask).item()

    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")

def main():
    args = parameter_parser()

    if args['random_seed'] is not None:
        random_seed = args['random_seed']
    else:
        random_seed = int(time.time())
    set_seed(random_seed)

    data_class = args['dataset_class']
    data = data_class()

    train_mask, val_mask, test_mask = split_data(data["num_vertices"], args['train_ratio'], args['val_ratio'], args['test_ratio'])

    forgotten_edge_list = forget_edges(data['edge_list'], args['forget_ratio'])

    start_sampling_time = time.time()
    sampled_edge_list, sampled_nodes = sample_edges({"num_vertices": data["num_vertices"], "edge_list": forgotten_edge_list}, args['alpha'], sample_ratio=0.25)
    end_sampling_time = time.time()
    sampling_time = end_sampling_time - start_sampling_time
    print(f"Sampling time: {sampling_time:.5f} seconds")

    G = Hypergraph(data["num_vertices"], sampled_edge_list)

    device = torch.device(args['device']) if args['device'] == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
    X, lbl, G, net, optimizer = initialize_model({"num_vertices": data["num_vertices"], "labels": data["labels"], "edge_list": sampled_edge_list, "num_classes": data["num_classes"]}, device, args['model_class'], args['learning_rate'], args['weight_decay'])

    start_training_time = time.time()
    print("Training sampled model...")
    best_state, best_epoch, total_training_time = train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, args['num_epochs'])
    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    print(f"Model training time: {training_time:.5f} seconds")

    loss, acc = test_model(net, best_state, X, G, lbl, test_mask)
    print(f"Sampled model final result: epoch: {best_epoch}")
    print(f"Sampled model test loss: {loss:.5f}, test accuracy: {acc:.5f}")
    print(f"Sampled model total training time: {total_training_time:.2f} seconds")

    print_statistics(train_mask, val_mask, test_mask)

    print(f"Number of hyperedges used for training: {len(sampled_edge_list)}")

if __name__ == "__main__":
    main()
