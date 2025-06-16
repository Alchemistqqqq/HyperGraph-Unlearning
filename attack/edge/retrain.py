from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
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
    return acc

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = torch.eye(data["num_vertices"]), data["labels"]
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
        with torch.no_grad():
            val_res = infer(net, X, G, lbl, val_mask)
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
    res = infer(net, X, G, lbl, test_mask, test=True)
    return res

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
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data_class = args['dataset_class']
    data = data_class()

    total_vertices = data["num_vertices"]
    train_mask, val_mask, test_mask = split_data(total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio'])

    # 从文件中加载原始模型的后验概率
    with open('../original_posteriors.pkl', 'rb') as f:
        original_posteriors = pickle.load(f)
    original_posteriors = torch.tensor(original_posteriors).to(device)

    new_edge_list, forget_indices = forget_edges(data['edge_list'], args['forget_ratio'])
    G_unlearned = Hypergraph(total_vertices, new_edge_list)

    X, lbl, G_unlearned, net, optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'], args['weight_decay'])
    best_state_unlearned, best_epoch_unlearned, total_training_time_unlearned = train_and_evaluate_model(net, X, G_unlearned, lbl, train_mask, val_mask, optimizer, args['num_epochs'])

    unlearned_acc = test_model(net, best_state_unlearned, X, G_unlearned, lbl, test_mask)

    unlearned_posteriors = get_posteriors(net, X, G_unlearned)

    print(f"Unlearned model accuracy: {unlearned_acc:.5f}")

    evaluate_attack_performance(original_posteriors, unlearned_posteriors, train_mask, forget_indices, test_mask)

    print(f"Final result: best epoch: {best_epoch_unlearned}")
    print(f"Unlearned model training time: {total_training_time_unlearned:.2f} seconds")

    print_statistics(train_mask, val_mask, test_mask)

if __name__ == "__main__":
    main()
