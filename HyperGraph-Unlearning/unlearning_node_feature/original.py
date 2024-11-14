import time
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Hypergraph
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from parameter_parser import parameter_parser  

def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()

@torch.no_grad()
def infer(net, X, A, lbls, idx, evaluator, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        accuracy = evaluator.validate(lbls, outs)
    else:
        accuracy = evaluator.test(lbls, outs)
    return accuracy

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = data["features"], data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    return X, lbl, G, net, optimizer

def train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, num_epochs, evaluator):
    best_state = None
    best_epoch, best_val = 0, 0
    start_time = time.time()
    for epoch in range(num_epochs):
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_acc = infer(net, X, G, lbl, val_mask, evaluator)
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
    return best_state, best_epoch

def test_model(net, best_state, X, G, lbl, test_mask, evaluator):
    print("test...")
    net.load_state_dict(best_state)
    test_acc = infer(net, X, G, lbl, test_mask, evaluator, test=True)
    return test_acc

def print_statistics(train_mask, val_mask, test_mask):
    num_train = torch.sum(train_mask).item()
    num_val = torch.sum(val_mask).item()
    num_test = torch.sum(test_mask).item()
    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")


def main():
    args = parameter_parser()  

    random_seed = args['random_seed'] if args['random_seed'] is not None else 2024
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

    print("Training original model...")
    best_state, best_epoch = train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer,
                                                      args['num_epochs'], evaluator)

    print("Testing model...")
    test_acc = test_model(net, best_state, X, G, lbl, test_mask, evaluator)

    print(f"Final result: epoch: {best_epoch}, test results: {test_acc}")
    print_statistics(train_mask, val_mask, test_mask)


if __name__ == "__main__":
    main()

