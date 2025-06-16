from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
from sklearn.metrics import roc_auc_score  
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


def cosine_similarity_loss(original_embeddings, current_embeddings, node_mask):
    return 1 - F.cosine_similarity(original_embeddings[node_mask], current_embeddings[node_mask]).mean()

def train(net, X, G, lbl, train_mask, optimizer, epoch, device):
    net.train()
    start_time = time.time()
    optimizer.zero_grad()

    X, lbl = X.to(device), lbl.to(device)
    train_mask = train_mask.to(device)

    out = net(X, G)
    loss = F.cross_entropy(out[train_mask], lbl[train_mask])
    loss.backward()
    optimizer.step()
    
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch: {epoch}, Loss: {loss.item():.5f}, Time: {epoch_time:.5f}s")

def train_with_similarity_loss(net, X, G, lbl, train_mask, forget_mask, optimizer, epoch, device):
    net.train()
    start_time = time.time()
    optimizer.zero_grad()

    X, lbl = X.to(device), lbl.to(device)
    train_mask = train_mask.to(device)
    forget_mask = forget_mask.to(device)

    # 计算原始输出（注意：这里 X 已为遗忘后的特征矩阵）
    out = net(X, G)
    loss = F.cross_entropy(out[train_mask], lbl[train_mask])
    
    # 为遗忘节点计算相似性损失：期望其表示变化不大
    with torch.no_grad():
        original_embeddings = net(X, G)
    current_embeddings = net(X, G)
    similarity_loss = cosine_similarity_loss(original_embeddings, current_embeddings, forget_mask)
    
    total_loss = loss + similarity_loss * 0.5  # 相似性损失权重可调
    
    total_loss.backward()
    optimizer.step()
    
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch: {epoch}, Loss: {loss.item():.5f}, Similarity Loss: {similarity_loss.item():.5f}, Time: {epoch_time:.5f}s")

def infer(net, X, G, lbl, mask, device, test=False):
    net.eval()
    with torch.no_grad():
        X, lbl, mask = X.to(device), lbl.to(device), mask.to(device)
        out = net(X, G)
        loss = F.cross_entropy(out[mask], lbl[mask])
        pred = out[mask].max(1)[1]
        correct = pred.eq(lbl[mask]).sum().item()
        acc = correct / mask.sum().item()
    if test:
        print(f"Test set results: loss= {loss.item():.5f}, accuracy= {acc:.5f}")
    return acc

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    # 使用单位矩阵作为节点特征
    X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])

    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    return X, lbl, G, net, optimizer

def train_and_evaluate_model(net, X, G, lbl, train_mask, val_mask, optimizer, num_epochs, device):
    best_state = None
    best_epoch, best_val = 0, 0

    start_time = time.time()

    for epoch in range(num_epochs):
        train(net, X, G, lbl, train_mask, optimizer, epoch, device)
        with torch.no_grad():
            val_res = infer(net, X, G, lbl, val_mask, device)
        if val_res > best_val:
            print(f"update best: {val_res:.5f}")
            best_epoch = epoch
            best_val = val_res
            best_state = deepcopy(net.state_dict())

    end_time = time.time()
    total_training_time = end_time - start_time

    print("\nTraining finished!")
    print(f"Best val accuracy: {best_val:.5f}")
    print(f"Total training time: {total_training_time:.2f} seconds")

    return best_state, best_epoch, total_training_time

def train_and_evaluate_unlearning(net, X, G, lbl, train_mask, forget_mask, val_mask, optimizer, num_epochs, device):

    best_state = None
    best_epoch, best_val = 0, 0

    start_time = time.time()

    for epoch in range(num_epochs):
        train_with_similarity_loss(net, X, G, lbl, train_mask, forget_mask, optimizer, epoch, device)
        with torch.no_grad():
            val_res = infer(net, X, G, lbl, val_mask, device)
        if val_res > best_val:
            print(f"update best: {val_res:.5f}")
            best_epoch = epoch
            best_val = val_res
            best_state = deepcopy(net.state_dict())

    end_time = time.time()
    total_training_time = end_time - start_time

    print("\nUnlearning finished!")
    print(f"Best val accuracy: {best_val:.5f}")
    print(f"Total unlearning time: {total_training_time:.2f} seconds")

    return best_state, best_epoch, total_training_time

def test_model(net, best_state, X, G, lbl, test_mask, device):
    print("\nTesting model...")
    net.load_state_dict(best_state)
    res = infer(net, X, G, lbl, test_mask, device, test=True)
    return res

def print_statistics(train_mask, val_mask, test_mask):
    num_train = int(torch.sum(train_mask).item())
    num_val = int(torch.sum(val_mask).item())
    num_test = int(torch.sum(test_mask).item())

    print("\n=== Dataset Statistics ===")
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
    
    device = torch.device(args['device'] if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data_class = args['dataset_class']
    data = data_class()
    total_vertices = data["num_vertices"]
    
    train_mask, val_mask, test_mask = split_data(
        total_vertices, 
        args['train_ratio'], 
        args['val_ratio'], 
        args['test_ratio']
    )
    
    X, lbl, G, net, optimizer = initialize_model(
        data, device, args['model_class'], 
        args['learning_rate'], args['weight_decay']
    )  
    print("\n=== Training Original Model ===")
    best_state_original, _, _ = train_and_evaluate_model(
        net, X, G, lbl, train_mask, val_mask, 
        optimizer, args['num_epochs'], device
    )

    with open('../original_posteriors.pkl', 'rb') as f:
        original_posteriors = torch.tensor(pickle.load(f)).to(device)
    
    print("\n=== Performing Node Feature Forgetting ===")
    X_unlearn, forget_indices = forget_node_features(X, train_mask, args['forget_ratio'])
    forget_mask = torch.zeros(X.shape[0], dtype=torch.bool)
    forget_mask[forget_indices] = True
    
    best_state_unlearned, _, unlearn_time = train_and_evaluate_unlearning(
        net, X_unlearn, G, lbl, train_mask, forget_mask, 
        val_mask, optimizer, args['num_epochs'], device
    )
    
    unlearned_posteriors = get_posteriors(net, X_unlearn, G)
    
    unlearned_acc = test_model(net, best_state_unlearned, X_unlearn, G, lbl, test_mask, device)
    
    print("\n=== Membership Inference Attack Evaluation ===")
    evaluate_attack_performance(
        original_posteriors, 
        unlearned_posteriors,
        train_mask,        
        forget_indices.cpu(),    
        test_mask         
    )
    

    print_statistics(train_mask, val_mask, test_mask)
    print("\nFinal Results:")
    print(f"Unlearned model accuracy: {unlearned_acc:.5f}")
    print(f"Unlearning time: {unlearn_time:.2f} seconds")

if __name__ == "__main__":
    main()
