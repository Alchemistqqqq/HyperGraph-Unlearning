from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import random
from dhg import Hypergraph
from pure_eval import Evaluator
from parameter_parser import parameter_parser 

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
    return new_edge_list

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
    return pred, total_train_time, best_val

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
    num_high_degree_edges = int(num_edges * 0.2)
    high_degree_edges = [edge_list[i] for i, _ in edge_degrees[:num_high_degree_edges]]
    remaining_edges = [edge_list[i] for i, _ in edge_degrees[num_high_degree_edges:]]

    subgraphs = [[] for _ in range(num_subgraphs)]
    for subgraph in subgraphs:
        subgraph.extend(high_degree_edges)

    for i, edge in enumerate(remaining_edges):
        subgraphs[i % num_subgraphs].append(edge)

    subgraph_hypergraphs = [Hypergraph(num_vertices, subgraph) for subgraph in subgraphs]

    return subgraph_hypergraphs, [len(subgraph) for subgraph in subgraphs]

def weighted_average(predictions, weights):
    weighted_preds = torch.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        weighted_preds += pred * weight
    final_predictions = torch.argmax(weighted_preds, dim=1)
    return final_predictions

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

    original_posteriors = get_posteriors(net, X, G).to(device)

    new_edge_list = forget_edges(data['edge_list'], args['forget_ratio'])
    G = Hypergraph(total_vertices, new_edge_list).to(device)

    subgraphs, subgraph_edge_counts = split_into_subgraphs(new_edge_list, args['num_subgraphs'], total_vertices)

    all_predictions = []
    all_weights = []
    subgraph_train_times = []

    start_time = time.time()
    for i, subgraph in enumerate(subgraphs):
        print(f"Scheduling training for subgraph {i + 1}")
        subgraph_data = (i, subgraph.to(device), X, lbl, train_mask.to(device), val_mask.to(device), test_mask.to(device), device, args['model_class'], args['learning_rate'], args['weight_decay'], args['num_epochs'])
        pred, train_time, best_val = train_and_evaluate_subgraph(subgraph_data)
        all_predictions.append(pred)
        all_weights.append(best_val)
        subgraph_train_times.append(train_time)
    end_time = time.time()

    final_predictions = weighted_average(all_predictions, all_weights)

    correct = final_predictions.eq(lbl[test_mask]).sum().item()
    final_accuracy = correct / test_mask.sum().item()

    print(f"Final Test Accuracy after weighted averaging: {final_accuracy:.5f}")

    print_statistics(train_mask, val_mask, test_mask)
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    for i, (edge_count, train_time) in enumerate(zip(subgraph_edge_counts, subgraph_train_times)):
        print(f"Subgraph {i + 1} edge count: {edge_count}, training time: {train_time:.2f} seconds")

    current_posteriors = get_posteriors(net, X, G).to(device)

    evaluate_attack_performance(original_posteriors, current_posteriors, train_mask.to(device), test_mask.to(device))

if __name__ == "__main__":
    main()
