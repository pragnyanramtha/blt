# blt_entropy_model.py
import spacy
import numpy as np
from collections import deque

class BLTInspiredEntropyWalker:
    """
    An advanced walker inspired by the contextual entropy concepts in modern
    sequence models like BLT. It uses a dynamic, path-aware entropy threshold
    to detect sentence boundaries.
    """
    def __init__(self, graph, **kwargs):
        """
        Initializes the walker with a set of tunable hyperparameters.
        """
        self.graph = graph
        # Model Hyperparameters
        self.local_weight = kwargs.get('local_weight', 0.4)
        self.degree_penalty_factor = kwargs.get('degree_penalty_factor', 20.0)
        self.warmup_steps = kwargs.get('warmup_steps', 2)
        self.warmup_threshold = kwargs.get('warmup_threshold', 0.45)
        self.entropy_multiplier = kwargs.get('entropy_multiplier', 2.0)

        self.global_weight = 1.0 - self.local_weight
        self.nlp = spacy.load("en_core_web_lg")
        self._vector_cache = {}
        
        print("BLT-Inspired Entropy Walker Initialized with:")
        print(f"  - Local Coherence Weight: {self.local_weight}")
        print(f"  - Hub Node Penalty Factor: {self.degree_penalty_factor}")
        print(f"  - Warm-up Steps: {self.warmup_steps} (using static threshold: {self.warmup_threshold})")
        print(f"  - Dynamic Entropy Multiplier: {self.entropy_multiplier}")

    def _get_vector(self, text):
        if text not in self._vector_cache:
            doc = self.nlp(text)
            vec = doc.vector if doc.has_vector else np.zeros(self.nlp.vocab.vectors_length)
            self._vector_cache[text] = vec.get() if hasattr(vec, 'get') else vec
        return self._vector_cache[text]

    def _calculate_similarity(self, vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0): return 0.0
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _calculate_final_score(self, current_node, neighbor_node, anchor_vector):
        """Calculates the blended, penalized coherence score for a potential step."""
        current_vec = self._get_vector(current_node)
        neighbor_vec = self._get_vector(neighbor_node)

        global_coherence = self._calculate_similarity(neighbor_vec, anchor_vector)
        local_coherence = self._calculate_similarity(neighbor_vec, current_vec)
        
        blended_score = (self.global_weight * global_coherence) + (self.local_weight * local_coherence)
        
        degree = self.graph.degree(neighbor_node)
        penalty = 1.0 - (np.log1p(degree) / self.degree_penalty_factor)
        final_score = blended_score * penalty
        
        return final_score

    def find_sentence_cluster(self, starting_nodes):
        if not starting_nodes: return {}, []

        # 1. Create the weighted semantic anchor (same as before)
        weighted_vectors, total_weight = [], 0
        for node_text in starting_nodes:
            node_doc = self.nlp(node_text, disable=["parser", "ner"])
            pos = node_doc[0].pos_ if node_doc else "NOUN"
            weight = 3.0 if pos in ["PROPN", "NOUN"] else 0.5 if pos == "PRON" else 1.0
            vec = self._get_vector(node_text)
            if not np.all(vec == 0):
                weighted_vectors.append(vec * weight)
                total_weight += weight
        
        if not weighted_vectors: return {}, []
        anchor_vector = np.sum(weighted_vectors, axis=0) / total_weight if total_weight > 0 else np.zeros(self.nlp.vocab.vectors_length)

        # 2. Setup for traversal with path memory
        sentence_cluster = {}
        crossed_boundary_edges = []
        queue = deque([(node, [self._calculate_final_score(node, node, anchor_vector)]) for node in starting_nodes])
        visited = set(starting_nodes)

        for node in starting_nodes:
            sentence_cluster[node] = 1.0 - self._calculate_similarity(self._get_vector(node), anchor_vector)

        # 3. Perform the dynamic entropy traversal
        while queue:
            current_node, path_scores = queue.popleft()
            
            # The current running average entropy of the path
            current_avg_score = np.mean(path_scores)

            edges_to_explore = list(self.graph.out_edges(current_node, data=True)) + \
                               list(self.graph.in_edges(current_node, data=True))

            for u, v, _ in edges_to_explore:
                neighbor_node = v if u == current_node else u

                if neighbor_node not in visited:
                    visited.add(neighbor_node)
                    
                    final_score = self._calculate_final_score(current_node, neighbor_node, anchor_vector)
                    entropy = max(0, 1.0 - final_score)

                    # --- DYNAMIC THRESHOLD LOGIC ---
                    # Is this a warm-up step?
                    if len(path_scores) < self.warmup_steps:
                        # Use the simple static threshold
                        decision_threshold = self.warmup_threshold
                        is_dynamic_stop = False
                    else:
                        # Use the dynamic threshold based on the path's history
                        decision_threshold = 1.0 - (current_avg_score * self.entropy_multiplier)
                        is_dynamic_stop = True

                    if final_score >= decision_threshold:
                        # Coherent step: continue
                        sentence_cluster[neighbor_node] = entropy
                        new_path_scores = path_scores + [final_score]
                        queue.append((neighbor_node, new_path_scores))
                    else:
                        # Boundary Detected: HALT
                        stop_reason = "Dynamic Entropy" if is_dynamic_stop else "Warm-up Filter"
                        crossed_boundary_edges.append({
                            "from": current_node,
                            "to": neighbor_node,
                            "final_score": final_score,
                            "threshold": decision_threshold,
                            "reason": stop_reason
                        })
                        
        return sentence_cluster, crossed_boundary_edges