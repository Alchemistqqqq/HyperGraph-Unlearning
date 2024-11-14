import time
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Hypergraph
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from parameter_parser import parameter_parser  

def train(net, X, A, lbls, train_idx, optimizer, epoch, device):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    X, A, lbls = X.to(device), A.to(device), lbls.to(device)
    train_idx = train_idx.to(device)
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()

@torch.no_grad()
def infer(net, X, A, lbls, idx, evaluator, test=False, device='cpu'):
    net.eval()
    X, A, lbls = X.to(device), A.to(device), lbls.to(device)
    idx = idx.to(device)
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    accuracy = evaluator.validate(lbls, outs) if not test else evaluator.test(lbls, outs)
    return accuracy

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = data["features"].to(device), data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"]).to(device)

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return X, lbl, G, net, optimizer

def sample_top_20_percent_edges(edge_list, num_vertices):
    edge_sizes = [(i, len(edge)) for i, edge in enumerate(edge_list)]
    edge_sizes.sort(key=lambda x: x[1], reverse=True)
    num_top_edges = int(0.4 * len(edge_list))
    top_edges = [edge_list[i[0]] for i in edge_sizes[:num_top_edges]]
    sampled_hypergraph = Hypergraph(num_vertices, top_edges)
    return sampled_hypergraph

def train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, num_epochs, evaluator, device):
    best_state = None
    best_epoch, best_val = 0, 0
    start_time = time.time()
    for epoch in range(num_epochs):
        train(net, X, G, lbl, train_mask, optimizer, epoch, device)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_acc = infer(net, X, G, lbl, val_mask, evaluator, device=device)
            if val_acc > best_val:
                print(f"update best: {val_acc:.5f}")
                best_epoch = epoch
                best_val = val_acc
                best_state = deepcopy(net.state_dict())
    end_time = time.time()
    total_training_time = end_time - start_time
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    print(f"total training time: {total_training_time:.2f} seconds")
    return best_state, best_epoch, total_training_time

def test_model(net, best_state, X, G, lbl, test_mask, evaluator, device):
    print("test...")
    net.load_state_dict(best_state)
    test_acc = infer(net, X, G, lbl, test_mask, evaluator, test=True, device=device)
    return test_acc['accuracy']  
def print_statistics(train_mask, val_mask, test_mask):
    num_train = torch.sum(train_mask).item()
    num_val = torch.sum(val_mask).item()
    num_test = torch.sum(test_mask).item()
    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")

def print_summary(original_features, deleted_features, remaining_features, total_training_time, acc):
    print("\n--- Summary ---")
    print(f"Original features: {original_features}")
    print(f"Deleted features: {deleted_features}")
    print(f"Remaining features: {remaining_features}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Final test accuracy: {acc:.5f}")  

def main():
    args = parameter_parser()  

    if args['random_seed'] is not None:
        random_seed = args['random_seed']
    else:
        random_seed = int(time.time())
    set_seed(random_seed)

    device = torch.device(args['device'] if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data_class = args['dataset_class']
    data = data_class()

    total_vertices = data["num_vertices"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    X, lbl, G, net, optimizer = initialize_model(data, device, args['model_class'], args['learning_rate'],
                                                 args['weight_decay'])

    original_features = X.numel()

    sampled_hypergraph = sample_top_20_percent_edges(data['edge_list'], total_vertices)
    sampled_hypergraph = sampled_hypergraph.to(device)  

    print("Training model on top 20% sampled edges...")
    best_state, best_epoch, total_training_time = train_and_evaluate_model(net, X, sampled_hypergraph, lbl, train_mask, val_mask, optimizer,
                                                                           args['num_epochs'], evaluator, device)

    print("Testing model...")
    test_acc = test_model(net, best_state, X, sampled_hypergraph, lbl, test_mask, evaluator, device)

    print_summary(original_features, 0, (X != 0).sum().item(), total_training_time, test_acc)
    print_statistics(train_mask, val_mask, test_mask)

if __name__ == "__main__":
    main()
