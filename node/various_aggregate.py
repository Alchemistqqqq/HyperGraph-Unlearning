from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import random
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
    random.seed(seed)

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
        # 返回 logits（后续聚合时有的需要先 argmax 得到预测标签）
        pred_logits = out[mask]
        correct = pred_logits.max(1)[1].eq(lbl[mask]).sum().item()
        acc = correct / mask.sum().item()
    if test:
        print(f"Test set results: loss= {loss.item():.5f}, accuracy= {acc:.5f}")
    return acc, pred_logits

def initialize_model(data, device, model_class, learning_rate, weight_decay):
    X, lbl = torch.eye(data["num_vertices"]).to(device), data["labels"].to(device)
    G = Hypergraph(data["num_vertices"], data["edge_list"]).to(device)
    net = model_class(X.shape[1], 32, data["num_classes"], use_bn=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return X, lbl, G, net, optimizer

def train_and_evaluate_subgraph(subgraph_data):
    subgraph_index, subgraph, X, lbl, train_mask, val_mask, test_mask, device, model_class, learning_rate, weight_decay, num_epochs = subgraph_data

    sub_net = model_class(X.shape[1], 32, lbl.max().item() + 1, use_bn=True).to(device)
    sub_optimizer = optim.Adam(sub_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val = 0
    total_train_time = 0

    for epoch in range(num_epochs):
        start = time.time()
        train(sub_net, X, subgraph, lbl, train_mask, sub_optimizer, epoch, device)
        end = time.time()
        total_train_time += end - start
        with torch.no_grad():
            val_acc, _ = infer(sub_net, X, subgraph, lbl, val_mask, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = deepcopy(sub_net.state_dict())

    sub_net.load_state_dict(best_state)
    test_acc, pred_logits = infer(sub_net, X, subgraph, lbl, test_mask, device, test=True)
    print(f"Subgraph {subgraph_index + 1} result: {test_acc}")
    return pred_logits.cpu(), total_train_time, test_acc

def print_statistics(train_mask, val_mask, test_mask):
    num_train = torch.sum(train_mask).item()
    num_val   = torch.sum(val_mask).item()
    num_test  = torch.sum(test_mask).item()
    print(f"Training nodes: {num_train}")
    print(f"Validation nodes: {num_val}")
    print(f"Test nodes: {num_test}")

def split_into_subgraphs(edge_list, num_subgraphs, num_vertices):
    edge_degrees = [(i, len(edge)) for i, edge in enumerate(edge_list)]
    edge_degrees.sort(key=lambda x: x[1], reverse=True)
    num_edges = len(edge_list)
    num_high_degree_edges = int(num_edges * 0.4)
    high_degree_edges = [edge_list[i] for i, _ in edge_degrees[:num_high_degree_edges]]
    remaining_edges = [edge_list[i] for i, _ in edge_degrees[num_high_degree_edges:]]

    subgraphs = [[] for _ in range(num_subgraphs)]
    for subgraph in subgraphs:
        subgraph.extend(high_degree_edges)
    for i, edge in enumerate(remaining_edges):
        subgraphs[i % num_subgraphs].append(edge)
    subgraph_hypergraphs = [Hypergraph(num_vertices, subgraph) for subgraph in subgraphs]
    return subgraph_hypergraphs, subgraphs

def calculate_node_coverage_weights(subgraph_edge_lists, total_vertices):
    subgraph_node_counts = []
    for subgraph_edge_list in subgraph_edge_lists:
        unique_nodes = set()
        for edge in subgraph_edge_list:
            unique_nodes.update(edge)
        subgraph_node_counts.append(len(unique_nodes))
    coverage_weights = [count / total_vertices for count in subgraph_node_counts]
    return coverage_weights

##############################
# 四种聚合方法
##############################

# 1. 加权平均聚合（直接对 logits 加权求和，再 argmax）
def weighted_average(predictions, weights):
    device = predictions[0].device
    weighted_preds = torch.zeros_like(predictions[0], dtype=torch.float).to(device)
    for pred, weight in zip(predictions, weights):
        pred = pred.float().to(device)
        weighted_preds += pred * weight
    final_predictions = torch.argmax(weighted_preds, dim=1)
    return final_predictions

# 2. 加权投票聚合：对每个子图的 logits 先 argmax 得到预测标签，再转为 one-hot，并乘以子图权重后累加
def weighted_average_voting(predictions, weights, num_classes):
    device = predictions[0].device
    num_nodes = predictions[0].shape[0]
    votes = torch.zeros((num_nodes, num_classes), device=device, dtype=torch.float)
    for pred_logits, weight in zip(predictions, weights):
        pred_labels = torch.argmax(pred_logits, dim=1)
        one_hot = F.one_hot(pred_labels, num_classes=num_classes).float()
        votes += one_hot * weight
    final_predictions = torch.argmax(votes, dim=1)
    return final_predictions

# 3. 多数投票聚合：先将每个子模型预测 logits 转为类别标签，再对各模型的预测取众数
def majority_voting(predictions_labels):
    stacked_preds = torch.stack(predictions_labels, dim=1)  # shape: (num_samples, n_models)
    majority_vote, _ = torch.mode(stacked_preds, dim=1)
    return majority_vote

# 4. 平均投票聚合（实现与多数投票类似，此处也采用取众数方式）
def average_voting(predictions_labels):
    stacked_preds = torch.stack(predictions_labels, dim=0)  # shape: (n_models, num_samples)
    # 这里采用 torch.mode 得到每个测试样本的平均投票结果
    final_preds, _ = torch.mode(stacked_preds, dim=0)
    return final_preds

##############################
# 主函数
##############################
def main():
    args = parameter_parser()

    if args['random_seed'] is not None:
        random_seed = args['random_seed']
    else:
        random_seed = int(time.time())
    set_seed(random_seed)

    # 使用同一CUDA设备
    aggregate_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = args['dataset_class']()

    total_vertices = data['num_vertices']
    total_edges = len(data['edge_list'])
    train_mask, val_mask, test_mask = split_data(total_vertices, args['train_ratio'], args['val_ratio'], args['test_ratio'])
    X, lbl, G, net, optimizer = initialize_model(data, aggregate_device, args['model_class'], args['learning_rate'], args['weight_decay'])

    original_train_nodes = torch.sum(train_mask).item()
    original_train_edges = total_edges

    new_train_mask, forgotten_indices = forget_nodes(total_vertices, train_mask, args['forget_ratio'])
    updated_edge_list, num_removed_edges = update_hypergraph(data['edge_list'], forgotten_indices)
    G = Hypergraph(total_vertices, updated_edge_list).to(aggregate_device)

    subgraphs, subgraph_edge_lists = split_into_subgraphs(updated_edge_list, args['num_subgraphs'], total_vertices)

    # 存储所有子图的预测 logits、训练时长以及测试准确率
    all_predictions = []
    subgraph_train_times = []
    subgraph_results = []  # 每个子图的测试准确率

    start_time = time.time()
    for i, subgraph in enumerate(subgraphs):
        subgraph_data = (i, 
                         subgraph.to(aggregate_device), 
                         X.to(aggregate_device), 
                         lbl.to(aggregate_device), 
                         new_train_mask.to(aggregate_device), 
                         val_mask.to(aggregate_device), 
                         test_mask.to(aggregate_device), 
                         aggregate_device, 
                         args['model_class'], 
                         args['learning_rate'], 
                         args['weight_decay'], 
                         args['num_epochs'])
        pred, train_time, sub_acc = train_and_evaluate_subgraph(subgraph_data)
        all_predictions.append(pred)
        subgraph_train_times.append(train_time)
        subgraph_results.append(sub_acc)
    end_time = time.time()

    # 找出测试准确率最高的子图
    best_subgraph_idx = max(range(len(subgraph_results)), key=lambda i: subgraph_results[i])
    best_subgraph_accuracy = subgraph_results[best_subgraph_idx]
    print(f"\nBest subgraph: Subgraph {best_subgraph_idx + 1} with test accuracy: {best_subgraph_accuracy:.5f}")

    # 计算每个子图的节点覆盖率权重
    node_coverage_weights = calculate_node_coverage_weights(subgraph_edge_lists, total_vertices)

    # 1. 使用 weighted_average 聚合（直接对 logits 加权求和后 argmax）
    final_pred_weighted_average = weighted_average(all_predictions, node_coverage_weights).to(aggregate_device)

    # 2. 使用 weighted_average_voting 聚合
    final_pred_weighted_average_voting = weighted_average_voting(all_predictions, node_coverage_weights, data['num_classes']).to(aggregate_device)

    # 为多数和平均投票聚合，将每个子模型 logits 转为预测标签（argmax）
    all_predictions_labels = [torch.argmax(pred, dim=1) for pred in all_predictions]
    # 3. 多数投票聚合
    final_pred_majority_voting = majority_voting(all_predictions_labels).to(aggregate_device)
    # 4. 平均投票聚合
    final_pred_average_voting = average_voting(all_predictions_labels).to(aggregate_device)

    # 分别计算各聚合方法的测试准确率
    test_idx = test_mask.to(aggregate_device)
    correct1 = final_pred_weighted_average.eq(lbl[test_idx]).sum().item()
    acc_weighted_average = correct1 / test_idx.sum().item()

    correct2 = final_pred_weighted_average_voting.eq(lbl[test_idx]).sum().item()
    acc_weighted_average_voting = correct2 / test_idx.sum().item()

    correct3 = final_pred_majority_voting.eq(lbl[test_idx]).sum().item()
    acc_majority_voting = correct3 / test_idx.sum().item()

    correct4 = final_pred_average_voting.eq(lbl[test_idx]).sum().item()
    acc_average_voting = correct4 / test_idx.sum().item()

    print("\n--- Summary ---")
    forgotten_nodes = len(forgotten_indices)
    remaining_train_nodes = torch.sum(new_train_mask).item()
    remaining_edges = len(updated_edge_list)
    print(f"Original training nodes: {original_train_nodes}")
    print(f"Original training edges: {original_train_edges}")
    print(f"Forgotten nodes: {forgotten_nodes}")
    print(f"Removed edges: {num_removed_edges}")
    print(f"Remaining training nodes: {remaining_train_nodes}")
    print(f"Remaining edges: {remaining_edges}")
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    print_statistics(new_train_mask, val_mask, test_mask)

    for i, (edge_list, train_time) in enumerate(zip(subgraph_edge_lists, subgraph_train_times)):
        print(f"Subgraph {i + 1} edge count: {len(edge_list)}, training time: {train_time:.2f} seconds")

    print(f"{best_subgraph_accuracy:.5f}")
    print("\n--- Aggregation Results ---")
    print(f"{acc_majority_voting:.5f}")
    print(f"{acc_average_voting:.5f}")
    print(f"{acc_weighted_average_voting:.5f}")
    print(f"{acc_weighted_average:.5f}")

if __name__ == "__main__":
    main()
