# gnn_trainer.py (Improved Version for Stable Training)

import pickle
import os
import torch
import spacy
import networkx as nx
import numpy as np
from tqdm import tqdm
import random

from node2vec import Node2Vec
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# --- 1. GNN Model Definition ---
# A more powerful architecture with dedicated projection heads
class GNNEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # GCN layers to learn node representations based on graph structure
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # --- NEW: Add projection heads ---
        # These linear layers will process the GCN output, which often improves performance.
        self.source_proj = torch.nn.Linear(out_channels, out_channels)
        self.dest_proj = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        # Pass data through GCN layers
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- 2. Main Training Logic ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading base KG from 'output/war_and_peace_kg.pkl'...")
    with open('output/war_and_peace_kg.pkl', 'rb') as f:
        data = pickle.load(f)
    graph = data['graph']
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    print("Step 1: Creating combined node features...")
    # (The feature creation part is the same)
    nlp = spacy.load("en_core_web_lg")
    spacy_embeds = np.array([nlp(node).vector for node in tqdm(nodes, desc="SpaCy Embeddings")])
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    n2v_model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node2vec_embeds = np.array([n2v_model.wv[node] for node in nodes])
    combined_features = np.concatenate([spacy_embeds, node2vec_embeds], axis=1)
    features_tensor = torch.tensor(combined_features, dtype=torch.float).to(device)
    print(f"Combined feature matrix created with shape: {features_tensor.shape}")

    print("Step 2: Creating labeled dataset of edges...")
    # (The dataset creation part is the same)
    positive_edges = [(node_to_idx[u], node_to_idx[v]) for u, v, data in graph.edges(data=True) if data.get('sentence_id') is not None]
    negative_edges = []
    all_nodes_set = set(nodes)
    while len(negative_edges) < len(positive_edges):
        u_node, v_node = random.sample(list(all_nodes_set), 2)
        if not graph.has_edge(u_node, v_node):
            negative_edges.append((node_to_idx[u_node], node_to_idx[v_node]))
    
    edge_index_list = positive_edges + negative_edges
    labels_list = [1] * len(positive_edges) + [0] * len(negative_edges)
    
    # Shuffle the dataset for better training
    shuffled_data = list(zip(edge_index_list, labels_list))
    random.shuffle(shuffled_data)
    edge_index_list, labels_list = zip(*shuffled_data)
    
    labeled_edges = torch.tensor(list(zip(*edge_index_list)), dtype=torch.long).to(device)
    labels = torch.tensor(labels_list, dtype=torch.float).to(device)
    
    # We need a separate edge_index for message passing that contains ALL graph edges
    all_graph_edges = torch.tensor(list(zip(*[(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()])), dtype=torch.long).to(device)

    print("Step 3: Training the GNN Edge Classifier...")
    model = GNNEdgeClassifier(in_channels=features_tensor.shape[1], hidden_channels=256, out_channels=128).to(device)
    
    # --- NEW: Lower learning rate for stability ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(100): # Train for more epochs with a lower learning rate
        model.train()
        optimizer.zero_grad()
        
        # Get node embeddings from the GCN layers
        node_embeds = model(features_tensor, all_graph_edges)
        
        # Get embeddings for the specific edges we have labels for
        source_indices = labeled_edges[0]
        dest_indices = labeled_edges[1]
        
        source_embeds = node_embeds[source_indices]
        dest_embeds = node_embeds[dest_indices]
        
        # --- NEW: Use the projection heads ---
        source_embeds = model.source_proj(source_embeds)
        dest_embeds = model.dest_proj(dest_embeds)
        
        # Dot product as a similarity score
        preds = (source_embeds * dest_embeds).sum(dim=1)
        
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        accuracy = ((preds > 0).float() == labels).float().mean()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:02d}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%')

    print("Step 4: Saving trained models and features to 'output'...")
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'gnn_model.pth'))
    torch.save(features_tensor, os.path.join(output_dir, 'node_features.pt'))
    torch.save(node_to_idx, os.path.join(output_dir, 'node_to_idx.pt'))
    
    print("Models and features saved successfully.")

if __name__ == '__main__':
    main()