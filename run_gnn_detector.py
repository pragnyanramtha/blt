# run_gnn_detector.py
import pickle
import argparse
import os
import random
import torch

# Import the new detector
from gnn_detector import GNNBoundaryDetector

# The evaluation function remains the same
def evaluate(detected_nodes, ground_truth_nodes):
    detected_set = set(detected_nodes.keys())
    ground_truth_set = set(ground_truth_nodes)
    true_positives = detected_set.intersection(ground_truth_set)
    false_positives = detected_set.difference(ground_truth_set)
    false_negatives = ground_truth_set.difference(detected_set)
    precision = len(true_positives) / len(detected_set) if len(detected_set) > 0 else 0
    recall = len(true_positives) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1, "fp": false_positives, "fn": false_negatives}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentence boundary detection using a trained GNN.")
    parser.add_argument("--kg_file", type=str, default="output/war_and_peace_kg.pkl")
    parser.add_argument("--model_file", type=str, default="output/gnn_model.pth")
    parser.add_argument("--features_file", type=str, default="output/node_features.pt")
    parser.add_argument("--node_map_file", type=str, default="output/node_to_idx.pt")
    parser.add_argument("--num_tests", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.75, help="GNN prediction threshold for in-sentence links.")
    
    args, unknown = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("--- Loading Data and Models ---")
    with open(args.kg_file, "rb") as f:
        data = pickle.load(f)
    graph = data["graph"]
    sentence_map = data["sentence_map"]
    
    # Initialize the GNN-based detector
    detector = GNNBoundaryDetector(
        graph, 
        args.model_file, 
        args.features_file, 
        args.node_map_file, 
        device,
        threshold=args.threshold
    )
    
    # Run tests
    valid_sentence_ids = [sid for sid, nodes in sentence_map.items() if nodes]
    test_sentence_ids = random.sample(valid_sentence_ids, args.num_tests)

    for i, sent_id in enumerate(test_sentence_ids):
        start_nodes = list(sentence_map[sent_id])
        ground_truth = sentence_map[sent_id]

        print(f"\n\n--- Test Case {i+1}/{args.num_tests} (Source Sentence ID: {sent_id}) ---")
        print(f"Starting Nodes: {start_nodes}")

        detected_cluster, boundary_crossings = detector.find_sentence_cluster(start_nodes)
        sorted_cluster = sorted(detected_cluster.items(), key=lambda item: item[1], reverse=True)

        print("\n[Output] Detected Sentence Cluster (Nodes sorted by entropy):")
        for node, entropy in sorted_cluster:
            print(f"  - Entropy: {entropy:.4f} | Node: '{node}'")
        
        if boundary_crossings:
            print("\n[Output] Traversal Halted at Boundaries (GNN Prediction < Threshold):")
            for edge in boundary_crossings[:5]:
                print(f"  - From '{edge['from']}' to '{edge['to']}' (Probability: {edge['probability']:.2f} < {args.threshold})")

        eval_results = evaluate(detected_cluster, ground_truth)
        print("\n[Evaluation]")
        print(f"  - Precision: {eval_results['precision']:.2f}, Recall: {eval_results['recall']:.2f}, F1-Score: {eval_results['f1']:.2f}")
        if eval_results['fp']: print(f"  - False Positives: {eval_results['fp']}")
        if eval_results['fn']: print(f"  - False Negatives: {eval_results['fn']}")