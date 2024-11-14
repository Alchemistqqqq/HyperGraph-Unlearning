import torch
from dhg.data import Cooking200, CoauthorshipCora, CoauthorshipDBLP, CocitationCora, CocitationCiteseer
import matplotlib.pyplot as plt
from collections import Counter


def plot_distribution(degrees, counts, title, xlabel, ylabel, color):
    plt.figure(figsize=(10, 6))
    plt.bar(degrees, counts, width=0.8, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(degrees)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_scatter(x, y, title, xlabel, ylabel, color):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color=color, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def calculate_vertex_degree_distribution(data):
    edge_list = data['edge_list']
    num_vertices = data['num_vertices']

    vertex_degrees = torch.zeros(num_vertices, dtype=torch.int)
    for edge in edge_list:
        for vertex in edge:
            vertex_degrees[vertex] += 1

    degree_count = Counter(vertex_degrees.tolist())
    degrees, counts = zip(*degree_count.items())

    return degrees, counts, vertex_degrees


def calculate_hyperedge_degree_distribution(data):
    edge_list = data['edge_list']

    edge_degrees = [len(edge) for edge in edge_list]
    degree_count = Counter(edge_degrees)
    degrees, counts = zip(*degree_count.items())

    return degrees, counts, edge_degrees


def main():
    datasets = [
        ("Cooking200", Cooking200(), 'skyblue'),
        # ("CoauthorshipCora", CoauthorshipCora(), 'orange'),
        # ("CoauthorshipDBLP", CoauthorshipDBLP(), 'green'),
        # ("CocitationCora", CocitationCora(), 'red'),
        # ("CocitationCiteseer", CocitationCiteseer(), 'purple')
    ]

    for name, data, color in datasets:
        vertex_degrees, vertex_counts, vertex_degrees_list = calculate_vertex_degree_distribution(data)
        hyperedge_degrees, hyperedge_counts, hyperedge_degrees_list = calculate_hyperedge_degree_distribution(data)

        plot_distribution(vertex_degrees, vertex_counts, f'Degree Distribution of Vertices in {name} Hypergraph',
                          'Degree of Vertices', 'Frequency', color)
        plot_distribution(hyperedge_degrees, hyperedge_counts,
                          f'Degree Distribution of Hyperedges in {name} Hypergraph', 'Degree of Hyperedges',
                          'Frequency', color)

        # Prepare data for scatter plot of vertex degree vs hyperedge degree
        edge_list = data['edge_list']
        vertex_hyperedge_pairs = []
        for edge in edge_list:
            hyperedge_degree = len(edge)
            for vertex in edge:
                vertex_hyperedge_pairs.append((vertex_degrees_list[vertex].item(), hyperedge_degree))

        vertex_degrees_scatter, hyperedge_degrees_scatter = zip(*vertex_hyperedge_pairs)
        plot_scatter(vertex_degrees_scatter, hyperedge_degrees_scatter,
                     f'Vertex Degree vs Hyperedge Degree in {name} Hypergraph',
                     'Vertex Degree', 'Hyperedge Degree', color)


if __name__ == "__main__":
    main()
