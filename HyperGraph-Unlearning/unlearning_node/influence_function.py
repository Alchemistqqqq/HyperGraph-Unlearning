from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dhg import Hypergraph
from pure_eval import Evaluator
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


def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = torch.eye(data["num_vertices"]).to(device), data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"]).to(device)

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return X, lbl, G, net, optimizer


def compute_gradients(model, X, G, lbl, mask):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer.zero_grad()
    out = model(X, G)
    loss = F.cross_entropy(out[mask], lbl[mask])
    loss.backward()
    gradients = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    return gradients


def hessian_vector_product(model, X, G, lbl, mask, vector, damping=0.01):
    model.train()
    out = model(X, G)
    loss = F.cross_entropy(out[mask], lbl[mask])
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    grad_product = sum((grad * vector[name]).sum() for name, grad in zip(vector.keys(), grads) if grad is not None)
    hessian_vector_grads = torch.autograd.grad(grad_product, model.parameters(), allow_unused=True)
    hvp = {name: hv_grad + damping * vector[name] for name, hv_grad in zip(vector.keys(), hessian_vector_grads) if hv_grad is not None}
    return hvp


def compute_influence_function(model, X, G, lbl, mask, grad_diff, scale=25, damping=0.01):
    v = deepcopy(grad_diff)
    for _ in range(scale):
        hvp = hessian_vector_product(model, X, G, lbl, mask, v, damping)
        v = {name: grad_diff[name] - hvp.get(name, torch.zeros_like(grad_diff[name])) for name in grad_diff.keys()}
    return v


def apply_gradients(model, grad_updates, scale=1.0):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in grad_updates:
                param.add_(grad_updates[name], alpha=-scale)


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

    device = torch.device(args['device']) if args['device'] == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    data_class = args['dataset_class']
    data = data_class()

    X, lbl, G, net, optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'], args['weight_decay'])

    total_vertices = data["num_vertices"]
    total_edges = data["num_edges"]
    train_mask, val_mask, test_mask = split_data(total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio'])

    original_train_nodes = torch.sum(train_mask).item()
    original_train_edges = total_edges

    best_state, best_epoch, total_training_time = train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, args['num_epochs'])

    print("Testing on original graph before applying influence...")
    loss_before, acc_before = test_model(net, best_state, X, G, lbl, test_mask)
    print(f"Result before influence: loss= {loss_before:.5f}, accuracy= {acc_before:.5f}")


    new_train_mask, forgotten_indices = forget_nodes(total_vertices, train_mask, args['forget_ratio'])

    updated_edge_list, num_removed_edges = update_hypergraph(data['edge_list'], forgotten_indices)

    G_remain = Hypergraph(data["num_vertices"], updated_edge_list).to(device)

    print("Calculating influence...")
    influence_start_time = time.time()

    train_grads = compute_gradients(net, X, G, lbl, train_mask)
    unlearn_grads = compute_gradients(net, X, G_remain, lbl, new_train_mask)  
    grad_diff = {name: train_grads[name] - unlearn_grads[name] for name in train_grads.keys()}

    influence = compute_influence_function(net, X, G, lbl, train_mask, grad_diff)
    influence_end_time = time.time()

    apply_gradients(net, influence, scale=args['apply_scale'])

    print("Testing after applying influence...")
    loss_after, acc_after = infer(net, X, G, lbl, test_mask, test=True) 
    print(f"Result after influence: loss= {loss_after:.5f}, accuracy= {acc_after:.5f}")

    total_end_time = time.time()
    total_time = total_end_time - influence_start_time
    influence_time = influence_end_time - influence_start_time

    print_statistics(new_train_mask, val_mask, test_mask)
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Influence calculation time: {influence_time:.2f} seconds")
    print(f"Overall time: {total_time:.2f} seconds")

    forgotten_nodes = len(forgotten_indices)
    remaining_train_nodes = torch.sum(new_train_mask).item()
    remaining_edges = len(updated_edge_list)

    print("\n--- Summary ---")
    print(f"Original training nodes: {original_train_nodes}")
    print(f"Original training edges: {original_train_edges}")
    print(f"Forgotten nodes: {forgotten_nodes}")
    print(f"Removed edges: {num_removed_edges}")
    print(f"Remaining training nodes: {remaining_train_nodes}")
    print(f"Remaining edges: {remaining_edges}")
    print(f"Final test accuracy before influence: {acc_before:.5f}")
    print(f"Final test accuracy after influence: {acc_after:.5f}")


if __name__ == "__main__":
    main()
