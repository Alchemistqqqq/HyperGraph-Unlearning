from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from dhg import Hypergraph
from sklearn.cluster import KMeans
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
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask)[1]
            print(f"Epoch: {epoch}, Validation accuracy: {val_res:.5f}")
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())

    end_time = time.time()
    total_training_time = end_time - start_time

    if best_state is None:
        best_state = deepcopy(net.state_dict())  # Use the last state if no improvement

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

def hypergraph_clique_expansion(edge_list, num_vertices):
    adj_matrix = np.zeros((num_vertices, num_vertices))
    for edge in edge_list:
        for i in range(len(edge)):
            for j in range(i+1, len(edge)):
                adj_matrix[edge[i], edge[j]] += 1
                adj_matrix[edge[j], edge[i]] += 1
    return adj_matrix

def cluster_hypergraph(edge_list, num_vertices, num_clusters):
    adj_matrix = hypergraph_clique_expansion(edge_list, num_vertices)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(adj_matrix)
    labels = kmeans.labels_
    return labels

def get_sub_hypergraph(edge_list, mask):
    sub_edge_list = [edge for edge in edge_list if all(mask[v] for v in edge)]
    sub_vertices = mask.sum().item()
    return sub_vertices, sub_edge_list

def partition_train(data, device, model_class, learning_rate, weight_decay, num_clusters, num_epochs):
    X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])

    labels = cluster_hypergraph(data["edge_list"], data["num_vertices"], num_clusters)
    partition_results = []

    for cluster_id in range(num_clusters):
        cluster_mask = torch.tensor(labels == cluster_id, dtype=torch.bool)
        train_mask, val_mask, test_mask = split_data(cluster_mask.sum().item(), 0.6, 0.2, 0.2)

        X_cluster, lbl_cluster = X[cluster_mask], lbl[cluster_mask]
        sub_vertices, sub_edge_list = get_sub_hypergraph(data["edge_list"], cluster_mask)
        G_cluster = Hypergraph(sub_vertices, sub_edge_list)

        print(f"Partition {cluster_id} - Number of hyperedges: {len(sub_edge_list)}")

        net = model_class(X_cluster.shape[1], 32, data["num_classes"], use_bn=True)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        X_cluster, lbl_cluster = X_cluster.to(device), lbl_cluster.to(device)
        G_cluster = G_cluster.to(device)
        net = net.to(device)

        best_state, best_epoch, total_training_time = train_and_evaluate_model(net, X_cluster, G_cluster, lbl_cluster, train_mask, val_mask, optimizer, num_epochs)
        if best_state is None:
            print(f"Warning: Partition {cluster_id} did not improve during training.")
        loss, acc = test_model(net, best_state, X_cluster, G_cluster, lbl_cluster, test_mask)
        partition_results.append((best_epoch, loss, acc, total_training_time))

    return partition_results

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

    print("Training partitioned model...")
    partition_results = partition_train(data, device, args['model_class'], args['learning_rate'], args['weight_decay'], 4, args['num_epochs'])  
    for i, (epoch, loss, acc, training_time) in enumerate(partition_results):
        print(f"Partition {i} - Epoch: {epoch}, Loss: {loss:.5f}, Accuracy: {acc:.5f}, Training time: {training_time:.2f} seconds")

if __name__ == "__main__":
    main()
