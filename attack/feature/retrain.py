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
    X, lbl = data["features"], data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    return X, lbl, G, net, optimizer


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

    X, forget_indices = forget_node_features(X, train_mask, args['forget_ratio'])

    best_state, best_epoch, total_training_time = train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, args['num_epochs'])

    loss, acc = test_model(net, best_state, X, G, lbl, test_mask)

    print(f"final result: epoch: {best_epoch}, loss: {loss}, accuracy: {acc}")

    print_statistics(train_mask, val_mask, test_mask)

    remaining_features = (X != 0).sum().item()

    with open('../original_posteriors.pkl', 'rb') as f:
        original_posteriors = pickle.load(f)
    original_posteriors = torch.tensor(original_posteriors).to(device)
    unlearned_posteriors = get_posteriors(net, X, G)
    evaluate_attack_performance(original_posteriors, unlearned_posteriors, train_mask, forget_indices.cpu(), test_mask)


if __name__ == "__main__":
    main()
