from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import sys
import os
from concurrent.futures import ProcessPoolExecutor

# Add this line to set the spawn method
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dhg import Hypergraph
from pure_eval import Evaluator
from parameter_parser import parameter_parser  

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

def update_hypergraph(edge_list, forgotten_indices):
    new_edge_list = []
    num_removed_edges = 0

    for edge in edge_list:
        updated_edge = [node for node in edge if node not in forgotten_indices]

        if len(updated_edge) > 0:
            new_edge_list.append(updated_edge)
        else:
            num_removed_edges += 1

    return new_edge_list, num_removed_edges

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
    return acc, out[mask]

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = torch.eye(data["num_vertices"]).to(device), data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"]).to(device)

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return X, lbl, G, net, optimizer

def train_and_evaluate_subgraph(subgraph_data):
    subgraph_index, subgraph, X, lbl, train_mask, val_mask, test_mask, device, model_class, learning_rate, weight_decay, num_epochs = subgraph_data

    sub_net = model_class(X.shape[1], 32, lbl.max().item() + 1, use_bn=True).to(device)
    sub_optimizer = optim.Adam(sub_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val = 0
    total_train_time = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        train(sub_net, X, subgraph, lbl, train_mask, sub_optimizer, epoch, device)
        end_time = time.time()
        total_train_time += end_time - start_time
        with torch.no_grad():
            val_res, _ = infer(sub_net, X, subgraph, lbl, val_mask, device)
        if val_res > best_val:
            best_val = val_res
            best_state = deepcopy(sub_net.state_dict())

    sub_net.load_state_dict(best_state)
    res, pred = infer(sub_net, X, subgraph, lbl, test_mask, device, test=True)

    print(f"Subgraph {subgraph_index + 1} result: {res}")
    return pred.cpu(), total_train_time, res  

def print_statistics(train_mask, val_mask, test_mask):
    num_train = torch.sum(train_mask).item()
    num_val = torch.sum(val_mask).item()
    num_test = torch.sum(test_mask).item()

    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")

def split_into_subgraphs(edge_list, num_subgraphs, num_vertices):
    edge_degrees = [(i, len(edge)) for i, edge in enumerate(edge_list)]
    edge_degrees.sort(key=lambda x: x[1], reverse=True)

    num_edges = len(edge_list)
    num_high_degree_edges = int(num_edges * 0.4)
    high_degree_edges = [edge_list[i] for i, _ in edge_degrees[:num_high_degree_edges]]
    remaining_edges = [edge_list[i] for i, _ in edge_degrees[num_high_degree_edges:]]

    subgraphs = [[] for _ in range(num_subgraphs)]
    for subgraph in subgraphs:
        subgraph.extend(high_degree_edges)

    for i, edge in enumerate(remaining_edges):
        subgraphs[i % num_subgraphs].append(edge)

    subgraph_hypergraphs = [Hypergraph(num_vertices, subgraph) for subgraph in subgraphs]

    return subgraph_hypergraphs, subgraphs

def calculate_node_coverage_weights(subgraph_edge_lists, total_vertices):
    subgraph_node_counts = []
    for subgraph_edge_list in subgraph_edge_lists:
        unique_nodes = set()
        for edge in subgraph_edge_list:
            unique_nodes.update(edge)
        subgraph_node_counts.append(len(unique_nodes))

    coverage_weights = [count / total_vertices for count in subgraph_node_counts]

    return coverage_weights

def weighted_average(predictions, weights):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weighted_preds = torch.zeros_like(predictions[0], dtype=torch.float).to(device)

    for pred, weight in zip(predictions, weights):
        pred = pred.float().to(device)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        weighted_preds += pred * weight

    if weighted_preds.dim() > 1 and weighted_preds.size(1) > 1:
        final_predictions = torch.argmax(weighted_preds, dim=1)
    else:
        final_predictions = torch.argmax(weighted_preds, dim=0)
    
    return final_predictions

def main():
    args = parameter_parser()

    if args['random_seed'] is not None:
        random_seed = args['random_seed']
    else:
        random_seed = int(time.time())
    set_seed(random_seed)

    aggregate_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data = args['dataset_class']()

    total_vertices = data['num_vertices']
    total_edges = len(data['edge_list'])
    train_mask, val_mask, test_mask = split_data(total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio'])

    X, lbl, G, net, optimizer = initialize_model(data, aggregate_device, args['model_class'], args['learning_rate'], args['weight_decay'])

    original_train_nodes = torch.sum(train_mask).item()
    original_train_edges = total_edges

    new_train_mask, forgotten_indices = forget_nodes(total_vertices, train_mask, args['forget_ratio'])

    updated_edge_list, num_removed_edges = update_hypergraph(data['edge_list'], forgotten_indices)
    G = Hypergraph(total_vertices, updated_edge_list).to(aggregate_device)

    subgraphs, subgraph_edge_lists = split_into_subgraphs(updated_edge_list, args['num_subgraphs'], total_vertices)

    all_predictions = []
    subgraph_train_times = []

    start_time = time.time()

    devices = [torch.device(f'cuda:{i}') for i in range(4)]  
    futures = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, subgraph in enumerate(subgraphs):
            subgraph_device = devices[i % len(devices)]
            subgraph_data = (i, subgraph.to(subgraph_device), X.to(subgraph_device), lbl.to(subgraph_device), 
                             new_train_mask.to(subgraph_device), val_mask.to(subgraph_device), test_mask.to(subgraph_device), 
                             subgraph_device, args['model_class'], args['learning_rate'], args['weight_decay'], args['num_epochs'])
            futures.append(executor.submit(train_and_evaluate_subgraph, subgraph_data))

        for future in futures:
            pred, train_time, _ = future.result()
            all_predictions.append(pred)
            subgraph_train_times.append(train_time)

    end_time = time.time()

    node_coverage_weights = calculate_node_coverage_weights(subgraph_edge_lists, total_vertices)
    final_predictions = weighted_average(all_predictions, node_coverage_weights).to(aggregate_device)

    correct = final_predictions.eq(lbl[test_mask].to(aggregate_device)).sum().item()
    final_accuracy = correct / test_mask.sum().item()

    print("\n--- Summary ---")
    forgotten_nodes = len(forgotten_indices)
    remaining_train_nodes = torch.sum(new_train_mask).item()
    remaining_edges = len(updated_edge_list)

    print(f"Original training nodes: {original_train_nodes}")
    print(f"Original training edges: {original_train_edges}")
    print(f"Forgotten nodes: {forgotten_nodes}")
    print(f"Removed edges: {num_removed_edges}")
    print(f"Remaining training nodes: {remaining_train_nodes}")
    print(f"Remaining edges: {remaining_edges}")
    print(f"Final test accuracy : {final_accuracy:.5f}")

    print_statistics(new_train_mask, val_mask, test_mask)
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    for i, (edge_list, train_time) in enumerate(zip(subgraph_edge_lists, subgraph_train_times)):
        print(f"Subgraph {i + 1} edge count: {len(edge_list)}, training time: {train_time:.2f} seconds")

if __name__ == "__main__":
    main()
