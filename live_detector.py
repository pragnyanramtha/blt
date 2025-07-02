# live_detector.py

import spacy
import networkx as nx
import numpy as np
from collections import defaultdict
import argparse

# Ensure you have the spaCy model downloaded:
# python -m spacy download en_core_web_lg

class SVO_KG_Builder:
    """Builds a simple SVO Knowledge Graph from a paragraph."""
    def __init__(self, model_name="en_core_web_lg"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Error: spaCy model '{model_name}' not found.")
            print("Please run: python -m spacy download en_core_web_lg")
            exit()
            
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer', first=True)

    def _get_clean_phrase(self, token):
        """Extracts a clean noun phrase from a token."""
        phrase_tokens = []
        for left in token.lefts:
            if left.dep_ in ('det', 'amod', 'compound', 'poss'):
                phrase_tokens.extend(list(left.subtree))
        
        phrase_tokens.append(token)
        phrase_tokens = sorted(list(set(phrase_tokens)), key=lambda t: t.i)
        
        final_tokens = []
        for t in phrase_tokens:
            if t.dep_ == 'cc': # Stop at coordinating conjunctions like 'and'
                break
            final_tokens.append(t)
        return " ".join(t.text for t in final_tokens).strip()

    def build(self, text):
        """Builds the graph from the input text."""
        doc = self.nlp(text)
        graph = nx.DiGraph()
        for sent in doc.sents:
            for token in sent:
                if token.pos_ in ("VERB", "AUX"):
                    subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                    if subjects and objects:
                        verb_phrase = self._get_clean_phrase(token)
                        for s_token in subjects:
                            s_phrase = self._get_clean_phrase(s_token)
                            for o_token in objects:
                                o_phrase = self._get_clean_phrase(o_token)
                                if s_phrase and o_phrase and verb_phrase:
                                    graph.add_node(s_phrase)
                                    graph.add_node(o_phrase)
                                    graph.add_edge(s_phrase, o_phrase, verb=verb_phrase)
        return graph

class LiveBoundaryDetector:
    """
    Traverses a KG from starting nodes and detects sentence boundaries
    based on semantic coherence. This is the model that performs the halting.
    """
    def __init__(self, graph, coherence_threshold=0.65):
        self.graph = graph
        self.threshold = coherence_threshold
        self.nlp = spacy.load("en_core_web_lg")
        self._vector_cache = {}

    def _get_vector(self, text):
        """Caches spaCy vectors for performance."""
        if text not in self._vector_cache:
            doc = self.nlp(text)
            self._vector_cache[text] = doc.vector if doc.has_vector else np.zeros(self.nlp.vocab.vectors_length)
        return self._vector_cache[text]

    def _calculate_similarity(self, vec1, vec2):
        """Calculates cosine similarity, handling potential zero vectors."""
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def find_sentence_cluster(self, starting_nodes):
        """
        Performs a traversal to find all nodes belonging to the source sentence.
        
        Returns:
            - sentence_cluster (dict): Nodes mapped to their entropy score.
            - crossed_boundary_edges (list): Edges that were rejected (the boundaries).
        """
        if not starting_nodes:
            return {}, []
            
        # Ensure all starting nodes are actually in the graph
        valid_start_nodes = [node for node in starting_nodes if node in self.graph]
        if not valid_start_nodes:
            print(f"Error: None of the starting nodes {starting_nodes} were found in the generated graph.")
            return {}, []

        # 1. Create the semantic anchor vector for the source sentence
        start_vectors = [self._get_vector(node) for node in valid_start_nodes]
        anchor_vector = np.mean([v for v in start_vectors if not np.all(v==0)], axis=0)

        # 2. Setup for traversal
        sentence_cluster = {}
        crossed_boundary_edges = []
        queue = list(valid_start_nodes)
        visited = set(valid_start_nodes)

        # Initialize starting nodes in the cluster with low entropy
        for node in valid_start_nodes:
            coherence = self._calculate_similarity(self._get_vector(node), anchor_vector)
            entropy = max(0, 1.0 - coherence)
            sentence_cluster[node] = entropy

        # 3. Perform the coherent traversal
        while queue:
            current_node = queue.pop(0)
            
            # Explore in ALL directions (outgoing and incoming edges)
            neighbors = set(self.graph.successors(current_node)) | set(self.graph.predecessors(current_node))
            
            for neighbor_node in neighbors:
                if neighbor_node in visited:
                    continue
                visited.add(neighbor_node)
                
                # THE CORE HALTING DECISION
                neighbor_vector = self._get_vector(neighbor_node)
                coherence_score = self._calculate_similarity(neighbor_vector, anchor_vector)

                if coherence_score >= self.threshold:
                    # Decision: Coherent. Stay within the sentence.
                    entropy = max(0, 1.0 - coherence_score)
                    sentence_cluster[neighbor_node] = entropy
                    queue.append(neighbor_node)
                else:
                    # Decision: Boundary Crossed. Halt traversal along this path.
                    crossed_boundary_edges.append({
                        "from": current_node,
                        "to": neighbor_node,
                        "coherence": coherence_score
                    })
                        
        return sentence_cluster, crossed_boundary_edges

def run_detection(paragraph, start_nodes_list, threshold):
    """Orchestrates the detection process and prints the results."""
    print("--- Input ---")
    print(f"Paragraph: \"{paragraph}\"")
    print(f"Starting Nodes: {start_nodes_list}")
    print(f"Coherence Threshold: {threshold}")
    
    # Build the KG from the paragraph
    kg_builder = SVO_KG_Builder()
    graph = kg_builder.build(paragraph)

    # Initialize and run the detector
    detector = LiveBoundaryDetector(graph, coherence_threshold=threshold)
    detected_cluster, boundary_crossings = detector.find_sentence_cluster(start_nodes_list)

    # Sort results by entropy for clear output
    sorted_cluster = sorted(detected_cluster.items(), key=lambda item: item[1], reverse=True)

    print("\n--- Output ---")
    if not sorted_cluster:
        print("Model did not find a coherent sentence cluster.")
        return
        
    print("\n[+] Detected Sentence Cluster (Nodes sorted by entropy):")
    for node, entropy in sorted_cluster:
        print(f"  - Entropy: {entropy:.4f} | Node: '{node}'")
    
    if boundary_crossings:
        print("\n[+] Traversal Halted at Boundaries:")
        for edge in boundary_crossings:
            print(f"  - From '{edge['from']}' to '{edge['to']}' (Coherence: {edge['coherence']:.2f} was below threshold)")


if __name__ == "__main__":
    # --- Example included directly in the code ---
    example_paragraph = (
        "The dreadful news of the battle of Borodinó, of our losses in killed and wounded, and the still more terrible news of the loss of Moscow reached Vorónezh in the middle of September Princess Mary, having learned of her brother’s wound only from the Gazette and having no definite news of him, prepared (so Nicholas heard, he had not seen her again himself) to set off in search of Prince Andrew"
    )
    example_start_nodes = ["computers", "vast amounts"]

    run_detection(example_paragraph, example_start_nodes, threshold=0.70)
    
    print("\n" + "="*80 + "\n")

    # --- Allow for custom input from the command line ---
    parser = argparse.ArgumentParser(
        description="Detect sentence boundaries in a custom paragraph using a KG and semantic coherence.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-p', '--paragraph', type=str, 
        help='The paragraph text to analyze.'
    )
    parser.add_argument(
        '-n', '--nodes', nargs='+', 
        help='A list of starting nodes (phrases) from the paragraph.'
    )
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.70,
        help='The semantic coherence threshold (0.0 to 1.0). Default is 0.70.'
    )
    
    args = parser.parse_args()

    if args.paragraph and args.nodes:
        print("Running with custom command-line input...")
        run_detection(args.paragraph, args.nodes, args.threshold)