# gnn_detector.py (Corrected Model Architecture)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import spacy
import numpy as np

# --- THE FIX IS HERE ---
# This class now EXACTLY matches the one used for training.
class GNNEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Add the projection heads that the trained model has
        self.source_proj = torch.nn.Linear(out_channels, out_channels)
        self.dest_proj = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict_edge(self, node_embeddings, source_idx, dest_idx):
        # This prediction logic now also uses the projection heads
        source_vec = self.source_proj(node_embeddings[source_idx])
        dest_vec = self.dest_proj(node_embeddings[dest_idx])
        
        pred = (source_vec * dest_vec).sum()
        return torch.sigmoid(pred)
# --- END OF FIX ---


class GNNBoundaryDetector:
    def __init__(self, graph, model_path, features_path, node_map_path, device, threshold=0.7):
        self.graph = graph
        self.device = device
        self.threshold = threshold
        
        print("Loading GNN model and features...")
        self.features = torch.load(features_path).to(self.device)
        self.node_to_idx = torch.load(node_map_path)
        
        feature_size = self.features.shape[1]
        
        # Initialize the model with the CORRECT architecture and hidden sizes
        self.model = GNNEdgeClassifier(
            in_channels=feature_size, 
            hidden_channels=256, # Must match trainer
            out_channels=128     # Must match trainer
        ).to(self.device)
        
        # This will now succeed because the architectures match
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        print("Pre-computing all node embeddings...")
        with torch.no_grad():
            edge_list = list(self.graph.edges())
            source_nodes = [self.node_to_idx[u] for u, v in edge_list if u in self.node_to_idx and v in self.node_to_idx]
            target_nodes = [self.node_to_idx[v] for u, v in edge_list if u in self.node_to_idx and v in self.node_to_idx]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to(self.device)
            
            self.all_node_embeddings = self.model(self.features, edge_index)
        print("Detector is ready.")

    def find_sentence_cluster(self, starting_nodes):
        if not all(node in self.node_to_idx for node in starting_nodes):
            print("Warning: Some starting nodes not found in graph. Skipping.")
            return {}, []
            
        sentence_cluster = {}
        crossed_boundary_edges = []
        queue = list(starting_nodes)
        visited = set(starting_nodes)

        for node in starting_nodes:
            sentence_cluster[node] = 0.05 # Low entropy for starting nodes

        while queue:
            current_node = queue.pop(0)
            neighbors = set(self.graph.successors(current_node)) | set(self.graph.predecessors(current_node))
            current_idx = self.node_to_idx[current_node]

            for neighbor_node in neighbors:
                if neighbor_node in visited or neighbor_node not in self.node_to_idx:
                    continue
                visited.add(neighbor_node)

                neighbor_idx = self.node_to_idx[neighbor_node]
                
                with torch.no_grad():
                    prob = self.model.predict_edge(self.all_node_embeddings, current_idx, neighbor_idx).item()
                
                if prob >= self.threshold:
                    entropy = 1.0 - prob
                    sentence_cluster[neighbor_node] = entropy
                    queue.append(neighbor_node)
                else:
                    crossed_boundary_edges.append({"from": current_node, "to": neighbor_node, "probability": prob})
                        
        return sentence_cluster, crossed_boundary_edges