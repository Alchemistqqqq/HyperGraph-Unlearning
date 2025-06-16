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

def forget_nodes(train_mask, forget_ratio):
    """节点遗忘函数"""
    train_indices = torch.where(train_mask)[0]
    num_forget = int(len(train_indices) * forget_ratio)
    forget_indices = train_indices[torch.randperm(len(train_indices))[:num_forget]]
    
    new_train_mask = train_mask.clone()
    new_train_mask[forget_indices] = False
    
    print(f"\n=== Node Forgetting Statistics ===")
    print(f"Original training nodes: {train_mask.sum().item()}")
    print(f"Forgotten nodes: {num_forget}")
    print(f"Remaining training nodes: {new_train_mask.sum().item()}")
    
    return new_train_mask, forget_indices

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
    """带相似性损失的节点遗忘训练"""
    net.train()
    start_time = time.time()
    optimizer.zero_grad()

    X, lbl = X.to(device), lbl.to(device)
    train_mask = train_mask.to(device)
    forget_mask = forget_mask.to(device)

    # 原始输出
    out_original = net(X, G)
    
    # 计算分类损失（仅使用剩余训练节点）
    loss = F.cross_entropy(out_original[train_mask], lbl[train_mask])
    
    # 相似性损失（确保遗忘节点的表示变化不大）
    with torch.no_grad():
        original_embeddings = net(X, G)
    
    current_embeddings = net(X, G)
    similarity_loss = cosine_similarity_loss(original_embeddings, current_embeddings, forget_mask)
    
    # 组合损失
    total_loss = loss + similarity_loss * 0.5  # 可调整权重
    
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
        if epoch % 1 == 0:
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
    """节点遗忘训练流程"""
    best_state = None
    best_epoch, best_val = 0, 0

    start_time = time.time()

    for epoch in range(num_epochs):
        train_with_similarity_loss(net, X, G, lbl, train_mask, forget_mask, optimizer, epoch, device)
        if epoch % 1 == 0:
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
    num_train = torch.sum(train_mask).item()
    num_val = torch.sum(val_mask).item()
    num_test = torch.sum(test_mask).item()

    print(f"\n=== Dataset Statistics ===")
    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")

def main():
    args = parameter_parser()

    # 设置随机种子
    if args['random_seed'] is not None:
        random_seed = args['random_seed']
    else:
        random_seed = int(time.time())
    set_seed(random_seed)
    
    # 设备设置
    device = torch.device(args['device'] if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    # 数据加载
    data_class = args['dataset_class']
    data = data_class()
    total_vertices = data["num_vertices"]
    
    # 划分数据集
    train_mask, val_mask, test_mask = split_data(
        total_vertices, 
        args['train_ratio'], 
        args['val_ratio'], 
        args['test_ratio']
    )
    
    # 初始化模型
    X, lbl, G, net, optimizer = initialize_model(
        data, device, args['model_class'], 
        args['learning_rate'], args['weight_decay']
    )
    
    # 原始模型训练
    print("\n=== Training Original Model ===")
    best_state_original, _, _ = train_and_evaluate_model(
        net, X, G, lbl, train_mask, val_mask, 
        optimizer, args['num_epochs'], device
    )
    
    # 获取原始后验概率
    #original_posteriors = get_posteriors(net, X, G)
    with open('../original_posteriors.pkl', 'rb') as f:
        original_posteriors = torch.tensor(pickle.load(f)).to(device)
    
    # 节点遗忘
    print("\n=== Performing Node Forgetting ===")
    new_train_mask, forget_indices = forget_nodes(train_mask, args['forget_ratio'])
    forget_mask = torch.zeros_like(train_mask)
    forget_mask[forget_indices] = True
    
    # 遗忘后训练
    best_state_unlearned, _, unlearn_time = train_and_evaluate_unlearning(
        net, X, G, lbl, new_train_mask, forget_mask, 
        val_mask, optimizer, args['num_epochs'], device
    )
    
    # 获取遗忘后后验概率
    unlearned_posteriors = get_posteriors(net, X, G)
    
    # 测试遗忘后模型
    unlearned_acc = test_model(net, best_state_unlearned, X, G, lbl, test_mask, device)
    
    # 成员推断攻击评估
    print("\n=== Membership Inference Attack Evaluation ===")
    evaluate_attack_performance(
        original_posteriors, 
        unlearned_posteriors,
        train_mask,        # Original full train mask
        forget_indices,    # Indices of forgotten nodes
        test_mask          # Test mask
    )
    
    # 输出统计信息
    print_statistics(new_train_mask, val_mask, test_mask)
    print(f"\nFinal Results:")
    print(f"Unlearned model accuracy: {unlearned_acc:.5f}")
    print(f"Unlearning time: {unlearn_time:.2f} seconds")

if __name__ == "__main__":
    main()