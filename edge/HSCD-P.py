from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import random
from concurrent.futures import ProcessPoolExecutor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dhg import Hypergraph
from pure_eval import Evaluator
from parameter_parser import parameter_parser  
import torch.multiprocessing as mp



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
    print(f"Epoch: {epoch}, Loss: {loss.item():.5f}, Time: {(end_time - start_time):.5f}s")


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
    #X = data["features"].to(device)
    #lbl = data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"]).to(device)

    net = model_class(X.shape[1], 16, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return X, lbl, G, net, optimizer


def train_and_evaluate_subgraph(subgraph_data):
    (
        subgraph_index,
        subgraph,       # this is a Hypergraph on CPU
        X,              # this is on the main device (cuda:0)
        lbl,            # on cuda:0
        train_mask,     # on CPU
        val_mask,       # on CPU
        test_mask,      # on CPU
        device,
        model_class,
        learning_rate,
        weight_decay,
        num_epochs
    ) = subgraph_data

    # ---- MOVE EVERYTHING TO THE CORRECT DEVICE ----
    subgraph = subgraph.to(device)
    X = X.to(device)
    lbl = lbl.to(device)
    train_mask = train_mask.to(device)
    val_mask   = val_mask.to(device)
    test_mask  = test_mask.to(device)
    # -----------------------------------------------

    sub_net = model_class(X.shape[1], 32, lbl.max().item() + 1, use_bn=True).to(device)
    sub_optimizer = optim.Adam(sub_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val = 0.0
    total_train_time = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train(sub_net, X, subgraph, lbl, train_mask, sub_optimizer, epoch, device)
        epoch_end = time.time()
        total_train_time += (epoch_end - epoch_start)

        with torch.no_grad():
            val_acc, _ = infer(sub_net, X, subgraph, lbl, val_mask, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = deepcopy(sub_net.state_dict())

    # Load best validation state
    sub_net.load_state_dict(best_state)

    # Final test
    test_acc, pred = infer(sub_net, X, subgraph, lbl, test_mask, device, test=True)
    print(f"Subgraph {subgraph_index + 1} result: {test_acc:.5f}")
    return pred.cpu(), total_train_time, best_val


def print_statistics(train_mask, val_mask, test_mask):
    print(f"Training nodes:   {train_mask.sum().item()}")
    print(f"Validation nodes: {val_mask.sum().item()}")
    print(f"Test nodes:       {test_mask.sum().item()}")


def split_into_subgraphs(edge_list, num_subgraphs, num_vertices):
    edge_degrees = [(i, len(edge)) for i, edge in enumerate(edge_list)]
    edge_degrees.sort(key=lambda x: x[1], reverse=True)

    num_edges = len(edge_list)
    num_high_degree_edges = int(num_edges * 0.6)
    high_degree_edges = [edge_list[i] for i, _ in edge_degrees[:num_high_degree_edges]]
    remaining_edges   = [edge_list[i] for i, _ in edge_degrees[num_high_degree_edges:]]

    subgraphs = [[] for _ in range(num_subgraphs)]
    for s in subgraphs:
        s.extend(high_degree_edges)
    for i, edge in enumerate(remaining_edges):
        subgraphs[i % num_subgraphs].append(edge)

    # Return a list of Hypergraph objects (on CPU for now) and their raw edge lists
    return [Hypergraph(num_vertices, s) for s in subgraphs], subgraphs


def calculate_node_coverage_weights(subgraphs, subgraph_edge_lists, total_vertices):
    counts = []
    for edge_list in subgraph_edge_lists:
        nodes = set()
        for e in edge_list:
            nodes.update(e)
        counts.append(len(nodes))
    weights = [c / total_vertices for c in counts]
    total = sum(weights)
    return [w / total for w in weights]


def weighted_average(predictions, weights):
    # predictions are CPU tensors
    weighted = torch.zeros_like(predictions[0], dtype=torch.float)
    for p, w in zip(predictions, weights):
        weighted += p.float() * w
    return weighted.argmax(dim=1)


def main():
    args = parameter_parser()
    mp.set_start_method('spawn')

    seed = args['random_seed'] if args['random_seed'] is not None else int(time.time())
    set_seed(seed)

    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data = args['dataset_class']()
    total_vertices = data['num_vertices']

    # Split masks on CPU
    train_mask, val_mask, test_mask = split_data(
        total_vertices,
        args['train_ratio'],
        args['val_ratio'],
        args['test_ratio']
    )

    # Initialize full model (only to get X, lbl); we won't train it end-to-end here
    X, lbl, G, net, optimizer = initialize_model(
        data,
        devices[0],
        args['model_class'],
        args['learning_rate'],
        args['weight_decay']
    )

    # Forget edges, rebuild G on device0
    new_edge_list = forget_edges(data['edge_list'], args['forget_ratio'])
    G = Hypergraph(total_vertices, new_edge_list).to(devices[0])

    # Split into subgraphs (still on CPU)
    subgraph_hgts, subgraph_edge_lists = split_into_subgraphs(
        new_edge_list,
        args['num_subgraphs'],
        total_vertices
    )

    all_preds = []
    train_times = []

    start = time.time()
    with ProcessPoolExecutor(max_workers=len(devices)) as exe:
        futures = []
        for i, sub_h in enumerate(subgraph_hgts):
            device = devices[i % len(devices)]
            print(f"Scheduling training for subgraph {i + 1} on {device}")
            payload = (
                i, sub_h, X, lbl,
                train_mask, val_mask, test_mask,
                device, args['model_class'],
                args['learning_rate'], args['weight_decay'],
                args['num_epochs']
            )
            futures.append(exe.submit(train_and_evaluate_subgraph, payload))

        for fut in futures:
            pred, ttime, _ = fut.result()
            all_preds.append(pred)
            train_times.append(ttime)
    end = time.time()

    weights = calculate_node_coverage_weights(subgraph_hgts, subgraph_edge_lists, total_vertices)
    final_pred = weighted_average(all_preds, weights)

    # Evaluate final
    correct = final_pred.eq(lbl.cpu()[test_mask]).sum().item()
    final_acc = correct / test_mask.sum().item()
    print(f"Final Test Accuracy after weighted averaging: {final_acc:.5f}")

    print_statistics(train_mask, val_mask, test_mask)
    print(f"Total training time: {(end - start):.2f}s")
    for i, (edges, ttime) in enumerate(zip(subgraph_edge_lists, train_times)):
        print(f"Subgraph {i+1}: {len(edges)} edges, {ttime:.2f}s training")


if __name__ == "__main__":
    main()  