# gnn_trainer.py
import pickle
import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, DataLoader

import spacy
from node2vec import Node2Vec

# --- 1. The GNN Model Definition ---
class EdgeClassifierGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Get contextualized node embeddings
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        
        # The classifier operates on pairs of nodes
        def classify(edge_list):
            src, dst = edge_list[0], edge_list[1]
            edge_features = torch.cat([h[src], h[dst]], dim=-1)
            return self.classifier(edge_features)
            
        return classify

# --- 2. Main Training Orchestrator ---
class GNNTrainer:
    def __init__(self, graph, sentence_map):
        self.graph = graph
        self.sentence_map = sentence_map
        self.nodes = list(graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def create_features(self):
        print("Step 1: Creating combined node features (Semantic + Structural)...")
        # a) Semantic features (spaCy)
        print("  - Generating spaCy embeddings...")
        nlp = spacy.load("en_core_web_lg")
        spacy_embeds = np.array([nlp(node).vector for node in tqdm(self.nodes)])

        # b) Structural features (Node2Vec)
        print("  - Generating Node2Vec embeddings...")
        node2vec = Node2Vec(self.graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
        n2v_model = node2vec.fit(window=10, min_count=1, batch_words=4)
        n2v_embeds = np.array([n2v_model.wv[node] for node in self.nodes])
        
        # c) Combine features
        self.features = np.concatenate([spacy_embeds, n2v_embeds], axis=1)
        self.n2v_model = n2v_model
        print(f"Combined feature matrix created with shape: {self.features.shape}")

    def create_labeled_dataset(self):
        print("Step 2: Creating labeled dataset of edges...")
        # Positive samples (all existing edges are within-sentence)
        positive_edges = []
        for u, v in self.graph.edges():
            positive_edges.append([self.node_to_idx[u], self.node_to_idx[v]])
        
        # Negative samples (random edges between nodes from different sentences)
        negative_edges = []
        sentence_nodes_list = list(self.sentence_map.values())
        num_neg_samples = len(positive_edges) * 2 # Create 2x negative samples
        
        print(f"  - Generating {num_neg_samples} negative samples...")
        with tqdm(total=num_neg_samples) as pbar:
            while len(negative_edges) < num_neg_samples:
                s1_nodes, s2_nodes = random.sample(sentence_nodes_list, 2)
                if not s1_nodes or not s2_nodes: continue
                u, v = random.choice(list(s1_nodes)), random.choice(list(s2_nodes))
                if u in self.node_to_idx and v in self.node_to_idx:
                    negative_edges.append([self.node_to_idx[u], self.node_to_idx[v]])
                    pbar.update(1)

        # Create labels (0 for positive, 1 for negative/boundary)
        pos_labels = torch.zeros(len(positive_edges))
        neg_labels = torch.ones(len(negative_edges))
        
        self.train_edges = torch.tensor(positive_edges + negative_edges).t().contiguous()
        self.train_labels = torch.cat([pos_labels, neg_labels], dim=0).long()

    def train(self, epochs=50):
        print("Step 3: Training the GNN Edge Classifier...")
        x = torch.tensor(self.features, dtype=torch.float).to(self.device)
        edge_index = torch.tensor(list(zip(*[(self.node_to_idx[u], self.node_to_idx[v]) for u,v in self.graph.edges()]))).to(self.device)
        
        model = EdgeClassifierGNN(self.features.shape[1], 128, 2).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            classifier = model(x, edge_index)
            # Get predictions for our labeled edges
            preds = classifier(self.train_edges.to(self.device))
            loss = F.cross_entropy(preds, self.train_labels.to(self.device))
            loss.backward()
            optimizer.step()
            
            # Simple accuracy check
            acc = ((preds.argmax(dim=-1) == self.train_labels.to(self.device)).sum() / len(self.train_labels)) * 100
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%')
            
        self.model = model

    def save_models(self, path="output"):
        print(f"Step 4: Saving trained models to '{path}'...")
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "gnn_model.pth"))
        self.n2v_model.save(os.path.join(path, "node2vec.model"))
        # Save the node mapping, it's crucial for inference
        with open(os.path.join(path, "node_mapping.pkl"), "wb") as f:
            pickle.dump({"nodes": self.nodes, "features": self.features}, f)
        print("Models saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GNN to detect sentence boundaries.")
    parser.add_argument("--kg_file", type=str, default="output/war_and_peace_kg.pkl")
    args, _ = parser.parse_known_args()

    print(f"Loading base KG from '{args.kg_file}'...")
    with open(args.kg_file, "rb") as f:
        data = pickle.load(f)
    graph, sentence_map = data["graph"], data["sentence_map"]

    trainer = GNNTrainer(graph, sentence_map)
    trainer.create_features()
    trainer.create_labeled_dataset()
    trainer.train(epochs=50) # You can adjust epochs
    trainer.save_models()