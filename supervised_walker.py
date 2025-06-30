# supervised_walker.py
import torch
import numpy as np
import spacy
from gensim.models import Word2Vec
import pickle
import os

# We need the GNN class definition to load the model
from gnn_trainer import EdgeClassifierGNN

class SupervisedBoundaryWalker:
    def __init__(self, graph, model_path="output"):
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading supervised models...")
        # Load node mapping and features
        with open(os.path.join(model_path, "node_mapping.pkl"), "rb") as f:
            map_data = pickle.load(f)
        self.nodes = map_data["features"]
        self.node_to_idx = {node: i for i, node in enumerate(map_data["nodes"])}
        
        # Load the trained GNN
        in_channels = self.nodes.shape[1]
        self.model = EdgeClassifierGNN(in_channels, 128, 2).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_path, "gnn_model.pth")))
        self.model.eval()

        # Pre-calculate contextual node embeddings for the whole graph
        print("Pre-calculating contextual node embeddings...")
        with torch.no_grad():
            x = torch.tensor(self.nodes, dtype=torch.float).to(self.device)
            edge_indices = list(zip(*[(self.node_to_idx[u], self.node_to_idx.get(v)) for u,v in graph.edges() if v in self.node_to_idx]))
            edge_index = torch.tensor(edge_indices).to(self.device) if edge_indices else torch.empty((2, 0), dtype=torch.long).to(self.device)
            # We call the forward pass up to the embeddings, not the final classifier
            self.contextual_embeds = self.model.conv2(self.model.conv1(x, edge_index).relu(), edge_index).relu()

        print("Supervised Walker ready.")

    def find_sentence_cluster(self, starting_nodes, halt_threshold=0.5):
        if not all(node in self.node_to_idx for node in starting_nodes):
            print("Warning: Some starting nodes not in graph. Aborting.")
            return {}, []
            
        sentence_cluster = {}
        crossed_boundary_edges = []
        queue = list(starting_nodes)
        visited = set(starting_nodes)

        for node in starting_nodes:
            sentence_cluster[node] = 0.0 # Starting nodes have zero entropy

        while queue:
            current_node = queue.pop(0)
            
            neighbors = set(self.graph.successors(current_node)) | set(self.graph.predecessors(current_node))
            
            for neighbor_node in neighbors:
                if neighbor_node in visited or neighbor_node not in self.node_to_idx:
                    continue
                visited.add(neighbor_node)

                # --- GNN-POWERED DECISION ---
                with torch.no_grad():
                    idx1, idx2 = self.node_to_idx[current_node], self.node_to_idx[neighbor_node]
                    
                    # Create edge feature from pre-calculated contextual embeddings
                    edge_feature = torch.cat([self.contextual_embeds[idx1], self.contextual_embeds[idx2]], dim=-1).unsqueeze(0)
                    
                    # Get prediction from the classifier head
                    logits = self.model.classifier(edge_feature)
                    probs = torch.softmax(logits, dim=1)
                    boundary_prob = probs[0][1].item() # Probability of class 1 (boundary)
                
                if boundary_prob < halt_threshold:
                    # Not a boundary, continue
                    sentence_cluster[neighbor_node] = boundary_prob # Entropy = boundary probability
                    queue.append(neighbor_node)
                else:
                    # Boundary detected, HALT
                    crossed_boundary_edges.append({
                        "from": current_node,
                        "to": neighbor_node,
                        "boundary_prob": boundary_prob
                    })
        
        return sentence_cluster, crossed_boundary_edges