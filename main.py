# main.py
import pickle, argparse, os, random

# Import our new supervised walker
from supervised_walker import SupervisedBoundaryWalker

# (The evaluate function remains the same as before)
def evaluate(detected_nodes, ground_truth_nodes):
    detected_set, ground_truth_set = set(detected_nodes.keys()), set(ground_truth_nodes)
    if not ground_truth_set:
        return {"precision": 0, "recall": 0, "f1": 0, "fp": detected_set, "fn": set()}
    tp = detected_set.intersection(ground_truth_set)
    fp = detected_set.difference(ground_truth_set)
    fn = ground_truth_set.difference(detected_set)
    precision = len(tp) / len(detected_set) if len(detected_set) > 0 else 0
    recall = len(tp) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1, "fp": fp, "fn": fn}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Supervised SBD on a pre-built KG.")
    parser.add_argument("--kg_file", type=str, default="output/war_and_peace_kg.pkl")
    parser.add_argument("--model_dir", type=str, default="output")
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--halt_threshold", type=float, default=0.5, help="Probability threshold to declare a boundary.")
    
    args, unknown = parser.parse_known_args()

    # Check if models are trained
    if not os.path.exists(os.path.join(args.model_dir, "gnn_model.pth")):
        print("Error: Trained GNN model not found.")
        print("Please run 'python gnn_trainer.py' first.")
        exit()

    print(f"Loading base KG from '{args.kg_file}'...")
    with open(args.kg_file, "rb") as f: data = pickle.load(f)
    graph, sentence_map = data["graph"], data["sentence_map"]

    # Instantiate the new Supervised walker
    detector = SupervisedBoundaryWalker(graph, model_path=args.model_dir)
    
    valid_ids = [sid for sid, nodes in sentence_map.items() if nodes]
    test_ids = random.sample(valid_ids, min(args.num_tests, len(valid_ids)))

    for i, sent_id in enumerate(test_ids):
        start_nodes, ground_truth = list(sentence_map[sent_id]), sentence_map[sent_id]
        
        print(f"\n\n--- Test Case {i+1}/{len(test_ids)} (Source Sentence ID: {sent_id}) ---")
        print(f"Starting Nodes: {start_nodes}")
        
        detected, boundaries = detector.find_sentence_cluster(start_nodes, halt_threshold=args.halt_threshold)
        sorted_cluster = sorted(detected.items(), key=lambda item: item[1], reverse=True)
        
        print("\n[Output] Detected Sentence Cluster (Sorted by Boundary Probability/Entropy):")
        for node, entropy in sorted_cluster: print(f"  - Entropy: {entropy:.4f} | Node: '{node}'")
        
        if boundaries:
            print("\n[Output] Traversal Halted at Boundaries:")
            for edge in boundaries[:5]:
                print(f"  - From '{edge['from']}' to '{edge['to']}' (Boundary Prob: {edge['boundary_prob']:.2f} >= {args.halt_threshold})")
        
        evals = evaluate(detected, ground_truth)
        print("\n[Evaluation]")
        print(f"  - Precision: {evals['precision']:.2f}, Recall: {evals['recall']:.2f}, F1-Score: {evals['f1']:.2f}")