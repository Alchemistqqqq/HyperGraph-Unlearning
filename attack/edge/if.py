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
    val_mask   = torch.zeros(total_vertices, dtype=torch.bool)
    test_mask  = torch.zeros(total_vertices, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices]     = True
    test_mask[test_indices]   = True

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
    epoch_time = time.time() - start_time
    print(f"Epoch: {epoch}, Loss: {loss.item():.5f}, Time: {epoch_time:.5f}s")

def infer(net, X, G, lbl, mask, test=False):
    net.eval()
    with torch.no_grad():
        out  = net(X, G)
        loss = F.cross_entropy(out[mask], lbl[mask])
        pred = out[mask].max(1)[1]
        correct = pred.eq(lbl[mask]).sum().item()
        acc = correct / mask.sum().item()
    if test:
        print(f"Test set results: loss= {loss:.5f}, accuracy= {acc:.5f}")
    return loss.item(), acc

def compute_gradients(model, X, G, lbl, mask, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer.zero_grad()
    out = model(X, G)
    loss = F.cross_entropy(out[mask], lbl[mask])
    loss.backward()
    grads = {
        name: param.grad.clone().to(device)
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return grads

def hessian_vector_product(model, X, G, lbl, mask, vector, device):
    model.train()
    out = model(X, G)
    loss = F.cross_entropy(out[mask], lbl[mask])
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    grad_product = sum(
        (grad * vector[name]).sum()
        for name, grad in zip(vector.keys(), grads)
        if grad is not None
    )
    hvp = torch.autograd.grad(grad_product, model.parameters(), allow_unused=True)
    return {
        name: hv.clone().to(device)
        for name, hv in zip(vector.keys(), hvp)
        if hv is not None
    }

def compute_influence_function(model, X, G, lbl, mask, grad_diff, device, damping=0.01, scale=25):
    v = deepcopy(grad_diff)
    for _ in range(scale):
        hvp = hessian_vector_product(model, X, G, lbl, mask, v, device)
        v = {
            name: grad_diff[name] - damping * hvp[name]
            for name in grad_diff.keys()
            if name in hvp
        }
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
    best_state, best_val = None, 0
    start_time = time.time()
    for epoch in range(num_epochs):
        train(net, X, G, lbl, train_mask.to(device), optimizer, epoch)
        val_acc = infer(net, X, G, lbl, val_mask.to(device))[1]
        if val_acc > best_val:
            best_val = val_acc
            best_state = deepcopy(net.state_dict())
            print(f"Update best model: val_acc={best_val:.5f}")
    total_time = time.time() - start_time
    print(f"\nTraining finished. Best val_acc={best_val:.5f}, time={total_time:.2f}s")
    return best_state, total_time

def test_model(net, best_state, X, G, lbl, test_mask, device):
    print("Testing model...")
    net.load_state_dict(best_state)
    return infer(net, X, G, lbl, test_mask.to(device), test=True)

def print_statistics(train_mask, val_mask, test_mask):
    print(f"Training nodes:   {train_mask.sum().item()}")
    print(f"Validation nodes: {val_mask.sum().item()}")
    print(f"Test nodes:       {test_mask.sum().item()}")

def main():
    args = parameter_parser()
    seed = args['random_seed'] or int(time.time())
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data = args['dataset_class']()
    total_vertices = data["num_vertices"]
    train_mask, val_mask, test_mask = split_data(
        total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio']
    )

    # 加载原始后验
    with open('../original_posteriors.pkl', 'rb') as f:
        original_posteriors = torch.tensor(pickle.load(f)).to(device)

    # 初始化模型和优化器
    X, lbl, G, net, optimizer = initialize_model(
        data, device, args['model_class'], args['learning_rate'], args['weight_decay']
    )

    # 忘记部分边，获取剩余边和遗忘边的索引
    remaining_edge_list, forget_indices = forget_edges(
        data['edge_list'], args['forget_ratio']
    )
    G_remain = Hypergraph(total_vertices, remaining_edge_list).to(device)

    # 在原图上训练并选出最佳模型
    best_state, train_time = train_and_evaluate_model(
        net, X, G, lbl, train_mask, val_mask, optimizer, args['num_epochs'], device
    )

    # 测试忘记前模型性能
    net.load_state_dict(best_state)
    print("Test before influence:")
    loss_before, acc_before = infer(net, X, G, lbl, test_mask.to(device), test=True)

    # 计算影响函数
    print("Calculating influence...")
    train_grads   = compute_gradients(net, X, G, lbl, train_mask, device)
    unlearn_grads = compute_gradients(net, X, G_remain, lbl, train_mask, device)
    grad_diff = {n: train_grads[n] - unlearn_grads[n] for n in train_grads}
    influence = compute_influence_function(
        net, X, G, lbl, train_mask, grad_diff, device
    )

    # 应用影响函数更新参数
    apply_gradients(net, influence, scale=args['apply_scale'])

    # 测试影响后模型性能
    print("Test after influence:")
    loss_after, acc_after = infer(net, X, G, lbl, test_mask.to(device), test=True)

    # 获取当前后验并评估攻击性能
    current_posteriors = get_posteriors(net, X, G).to(device)
    evaluate_attack_performance(
        original_posteriors,
        current_posteriors,
        train_mask.to(device),
        forget_indices,
        test_mask.to(device)
    )

    # 打印统计信息
    print_statistics(train_mask, val_mask, test_mask)
    print(f"Total training time: {train_time:.2f}s")

if __name__ == "__main__":
    main()
