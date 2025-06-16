from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
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

def forget_node_features(X, train_mask, forget_ratio):
    train_mask = train_mask.to(X.device)
    nonzero_idx = torch.nonzero(X).view(-1, 2)
    train_nonzero_mask = train_mask[nonzero_idx[:, 0]]
    nonzero_train = nonzero_idx[train_nonzero_mask] 
    total_nonzero_train = nonzero_train.shape[0]
    num_to_forget = int(total_nonzero_train * forget_ratio)
    if num_to_forget == 0:
        return X, torch.tensor([], dtype=torch.long)
    sel = nonzero_train[torch.randperm(total_nonzero_train)[:num_to_forget]]
    X[sel[:, 0], sel[:, 1]] = 0   
    forgotten_nodes = torch.unique(sel[:, 0])
    return X, forgotten_nodes

def train(net, X, G, lbl, train_mask, optimizer, epoch):
    net.train()
    start_time = time.time()
    optimizer.zero_grad()
    out = net(X, G)
    loss = F.cross_entropy(out[train_mask], lbl[train_mask])
    loss.backward()
    optimizer.step()
    end_time = time.time()
    print(f"Epoch: {epoch}, Loss: {loss.item():.5f}, Time: {end_time - start_time:.5f}s")

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

def compute_gradients(model, X, G, lbl, mask, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer.zero_grad()
    out = model(X, G)
    loss = F.cross_entropy(out[mask], lbl[mask])
    loss.backward()
    grads = {name: param.grad.clone().to(device) for name, param in model.named_parameters() if param.grad is not None}
    return grads

def hessian_vector_product(model, X, G, lbl, mask, vector, device):
    model.train()
    out = model(X, G)
    loss = F.cross_entropy(out[mask], lbl[mask])
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    grad_prod = sum((g * vector[name]).sum() for name, g in zip(vector.keys(), grads) if g is not None)
    hv_grads = torch.autograd.grad(grad_prod, model.parameters(), allow_unused=True)
    hvp = {name: hv.clone().to(device) for name, hv in zip(vector.keys(), hv_grads) if hv is not None}
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
    X = torch.eye(data["num_vertices"]).to(device)
    lbl = data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"]).to(device)

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return X, lbl, G, net, optimizer

def train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, num_epochs, device):
    best_state, best_val, best_epoch = None, 0, 0
    start_time = time.time()
    for epoch in range(num_epochs):
        train(net, X, G, lbl, train_mask.to(device), optimizer, epoch)
        _, val_acc = infer(net, X, G, lbl, val_mask.to(device))
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = deepcopy(net.state_dict())
            print(f"update best: {best_val:.5f}")
    total_time = time.time() - start_time
    print(f"\nTraining finished! Best val acc: {best_val:.5f} at epoch {best_epoch}")
    print(f"Total training time: {total_time:.2f}s")
    return best_state, best_epoch, total_time

def test_model(net, best_state, X, G, lbl, test_mask, device):
    net.load_state_dict(best_state)
    return infer(net, X, G, lbl, test_mask.to(device), test=True)

def print_statistics(train_mask, val_mask, test_mask, forget_indices=None):
    print(f"Training nodes: {train_mask.sum().item()}")
    print(f"Validation nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")
    if forget_indices is not None:
        print(f"Nodes with forgotten features: {len(forget_indices)}")

def main():
    args = parameter_parser()
    seed = args.get('random_seed') or int(time.time())
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data = args['dataset_class']()
    total_vertices = data["num_vertices"]
    train_mask, val_mask, test_mask = split_data(total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio'])

    with open('../original_posteriors.pkl', 'rb') as f:
        original_posteriors = torch.tensor(pickle.load(f)).to(device)

    X, lbl, G, net, optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'], args['weight_decay'])

    best_state, best_epoch, train_time = train_and_evaluate_model(
        net, X, G, lbl, train_mask, val_mask, optimizer, args['num_epochs'], device)
    
    net.load_state_dict(best_state)
    print("\nTest before applying influence:")
    loss_b, acc_b = infer(net, X, G, lbl, test_mask, test=True)


    train_grads = compute_gradients(net, X, G, lbl, train_mask, device)
    X_forgotten, forget_indices = forget_node_features(X, train_mask, args['forget_ratio'])
    unlearn_grads = compute_gradients(net, X_forgotten, G, lbl, train_mask, device)
    grad_diff = {k: train_grads[k] - unlearn_grads[k] for k in train_grads}

    print("\nCalculating influence...")
    inf_start = time.time()
    influence = compute_influence_function(net, X, G, lbl, train_mask, grad_diff, device)
    inf_time = time.time() - inf_start


    apply_gradients(net, influence, scale=args['apply_scale'])

    # Test after influence
    print("\nTest after applying influence:")
    loss_a, acc_a = infer(net, X, G, lbl, test_mask, test=True)

    current_posteriors = get_posteriors(net, X, G).to(device)
    evaluate_attack_performance(
        original_posteriors,
        current_posteriors,
        train_mask,
        forget_indices.cpu(),
        test_mask
    )

    print_statistics(train_mask, val_mask, test_mask, forget_indices)
    print(f"Total training time: {train_time:.2f}s")
    print(f"Influence calculation time: {inf_time:.2f}s")

if __name__ == "__main__":
    main()
