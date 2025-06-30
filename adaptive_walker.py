# adaptive_walker.py
import spacy
import numpy as np
from collections import deque

class AdaptiveEntropyWalker:
    """
    A state-of-the-art walker that adapts its behavior based on the
    quality of the semantic context it's exploring. It features an
    adaptive warm-up and a dynamic cluster centroid.
    """
    def __init__(self, graph, **kwargs):
        self.graph = graph
        # Hyperparameters
        self.local_weight = kwargs.get('local_weight', 0.4)
        self.base_warmup_threshold = kwargs.get('base_warmup_threshold', 0.40)
        self.entropy_multiplier = kwargs.get('entropy_multiplier', 2.0)
        self.recenter_interval = kwargs.get('recenter_interval', 5) # Recalculate anchor every 5 nodes

        self.global_weight = 1.0 - self.local_weight
        self.nlp = spacy.load("en_core_web_lg")
        self._vector_cache = {}
        
        print("Adaptive Entropy Walker Initialized.")

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

    def _calculate_score(self, current_node, neighbor_node, anchor_vector):
        """Calculates the blended coherence score for a potential step."""
        current_vec = self._get_vector(current_node)
        neighbor_vec = self._get_vector(neighbor_node)
        global_coherence = self._calculate_similarity(neighbor_vec, anchor_vector)
        local_coherence = self._calculate_similarity(neighbor_vec, current_vec)
        return (self.global_weight * global_coherence) + (self.local_weight * local_coherence)

    def find_sentence_cluster(self, starting_nodes):
        if not starting_nodes: return {}, []

        # 1. Create the initial anchor and assess its quality
        start_vectors = [self._get_vector(node) for node in starting_nodes]
        valid_start_vectors = [v for v in start_vectors if not np.all(v == 0)]
        if not valid_start_vectors: return {}, []
        
        anchor_vector = np.mean(valid_start_vectors, axis=0)
        # Measure anchor quality: high standard deviation = fuzzy anchor
        anchor_fuzziness = np.mean([np.std(v) for v in valid_start_vectors])
        
        # 2. ADAPTIVE WARM-UP THRESHOLD
        # If the anchor is fuzzy, use a much stricter threshold.
        adaptive_threshold = self.base_warmup_threshold - anchor_fuzziness
        print(f"  [Model] Anchor Fuzziness: {anchor_fuzziness:.3f}. Adaptive Warm-up Threshold: {adaptive_threshold:.3f}")

        # 3. Setup traversal
        sentence_cluster = {}
        crossed_boundary_edges = []
        # Queue: (node_to_visit, score_that_led_to_it)
        queue = deque([(node, 1.0) for node in starting_nodes]) # Start with perfect score
        visited = set(starting_nodes)
        
        nodes_since_recenter = 0

        while queue:
            current_node, entry_score = queue.popleft()
            
            # Add to cluster
            entropy = 1.0 - entry_score
            sentence_cluster[current_node] = entropy
            nodes_since_recenter += 1
            
            # 4. DYNAMIC CLUSTER CENTROID
            # Periodically recenter the anchor to lock onto the topic
            if nodes_since_recenter >= self.recenter_interval:
                print(f"  [Model] Recalculating cluster centroid...")
                cluster_vectors = [self._get_vector(node) for node in sentence_cluster.keys()]
                anchor_vector = np.mean(cluster_vectors, axis=0)
                nodes_since_recenter = 0

            # Running average of the SCORES of the path so far
            path_avg_score = np.mean([1.0-e for n,e in sentence_cluster.items()])

            edges_to_explore = list(self.graph.out_edges(current_node, data=True)) + \
                               list(self.graph.in_edges(current_node, data=True))

            for u, v, _ in edges_to_explore:
                neighbor_node = v if u == current_node else u
                if neighbor_node not in visited:
                    visited.add(neighbor_node)
                    
                    final_score = self._calculate_score(current_node, neighbor_node, anchor_vector)

                    # 5. DYNAMIC HALTING LOGIC
                    # Stop if the score is below the strict adaptive warm-up OR
                    # if the score is significantly worse than the path's average
                    dynamic_threshold = path_avg_score / self.entropy_multiplier
                    
                    if final_score >= adaptive_threshold and final_score >= dynamic_threshold:
                        queue.append((neighbor_node, final_score))
                    else:
                        stop_reason = "Dynamic Entropy" if final_score < dynamic_threshold else "Adaptive Warm-up"
                        crossed_boundary_edges.append({
                            "from": current_node,
                            "to": neighbor_node,
                            "score": final_score,
                            "threshold": max(adaptive_threshold, dynamic_threshold),
                            "reason": stop_reason
                        })
                        
        return sentence_cluster, crossed_boundary_edges