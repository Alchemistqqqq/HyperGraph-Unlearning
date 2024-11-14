from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

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
        print(f"Test set results: loss= {loss:.5f}, accuracy= {acc:.5f}")
    return loss, acc

def compute_gradients(model, X, G, lbl, mask, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer.zero_grad()
    out = model(X, G)
    loss = F.cross_entropy(out[mask], lbl[mask])
    loss.backward()
    gradients = {name: param.grad.clone().to(device) for name, param in model.named_parameters() if param.grad is not None}
    return gradients

def hessian_vector_product(model, X, G, lbl, mask, vector, device):
    model.train()
    out = model(X, G)
    loss = F.cross_entropy(out[mask], lbl[mask])
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    grad_product = sum((grad * vector[name]).sum() for name, grad in zip(vector.keys(), grads) if grad is not None)
    hessian_vector_grads = torch.autograd.grad(grad_product, model.parameters(), allow_unused=True)
    hvp = {name: hv_grad.clone().to(device) for name, hv_grad in zip(vector.keys(), hessian_vector_grads) if hv_grad is not None}
    return hvp

def compute_influence_function(model, X, G, lbl, mask, grad_diff, device, damping=0.01, scale=25):
    v = deepcopy(grad_diff)
    for _ in range(scale):
        hvp = hessian_vector_product(model, X, G, lbl, mask, v, device)
        v = {name: grad_diff[name] - damping * hvp[name] for name in grad_diff.keys() if name in hvp}
    return v

def apply_gradients(model, grad_updates, scale=1.0):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in grad_updates:
                param.add_(grad_updates[name], alpha=-scale)

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

def main():
    args = parameter_parser()

    if args['random_seed'] is not None:
        random_seed = args['random_seed']
    else:
        random_seed = int(time.time())
    set_seed(random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data_class = args['dataset_class']
    data = data_class()

    total_vertices = data["num_vertices"]
    train_mask, val_mask, test_mask = split_data(total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio'])

    X, lbl, G, net, optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'], args['weight_decay'])

    # unlearning edge
    remaining_edge_list = forget_edges(data['edge_list'], args['forget_ratio'])
    G_remain = Hypergraph(total_vertices, remaining_edge_list).to(device)

    total_start_time = time.time()

    best_state, best_epoch, total_training_time = train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, args['num_epochs'], device)

    net.load_state_dict(best_state)
    print("test before applying influence...")
    loss_before, acc_before = infer(net, X, G, lbl, test_mask, device, test=True)
    print(f"result before influence: loss= {loss_before:.5f}, accuracy= {acc_before:.5f}")

    print("Calculating influence...")
    influence_start_time = time.time()
    train_grads = compute_gradients(net, X, G, lbl, train_mask, device)
    unlearn_grads = compute_gradients(net, X, G_remain, lbl, train_mask, device)  
    grad_diff = {name: train_grads[name] - unlearn_grads[name] for name in train_grads.keys()}

    influence = compute_influence_function(net, X, G, lbl, train_mask, grad_diff, device)
    influence_end_time = time.time()

    apply_gradients(net, influence, scale=args['apply_scale'])

    print("Test after applying influence...")
    loss_after, acc_after = infer(net, X, G, lbl, test_mask, device, test=True)
    print(f"result after influence: loss= {loss_after:.5f}, accuracy= {acc_after:.5f}")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    influence_time = influence_end_time - influence_start_time

    print_statistics(train_mask, val_mask, test_mask)
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Influence calculation time: {influence_time:.2f} seconds")
    print(f"Overall time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
