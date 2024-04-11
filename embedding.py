import json
from matplotlib import pyplot as plt
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict

with open('./result_json.json', 'r') as file:
    graph_data = json.load(file)

def build_graph(graph_data: Dict):
    G = nx.Graph()
    for edge in graph_data:
        source = edge['source']
        target = edge['target']
        weight = edge['weight']
        G.add_edge(source, target, weight=weight)

def get_graph_attr(graph_data: Dict):
    pass
    
def graph_embed(G: nx.Graph):
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
    node_embeddings = np.array(list(embeddings.values()))
    graph_embedding = np.mean(node_embeddings, axis=0)
    print("图的嵌入向量:", graph_embedding)
    model.save('./graph_embeddings.model')
    # clustering analysis
    Z = linkage(graph_embedding, method='ward')
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Graph Embeddings')
    plt.ylabel('Distance')
    dendrogram(Z)
    plt.savefig("./ca.png")
