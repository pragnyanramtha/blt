# advanced_detector.py

import spacy
import numpy as np

class AdvancedBoundaryDetector:
    """
    An advanced model that traverses a KG using a blend of global, local,
    and structural coherence to find sentence boundaries.
    """
    def __init__(self, graph, global_threshold=0.65, local_weight=0.3, degree_penalty_factor=10):
        """
        Initializes the detector with tunable hyperparameters.
        
        Args:
            graph (nx.DiGraph): The SVO knowledge graph.
            global_threshold (float): The main coherence score needed to continue traversal.
            local_weight (float): How much to value local (node-to-node) coherence.
                                  (0.0 to 1.0). Global anchor weight will be (1.0 - local_weight).
            degree_penalty_factor (int): How sharply to penalize high-degree hub nodes.
                                         Higher value = smaller penalty.
        """
        self.graph = graph
        self.threshold = global_threshold
        self.local_weight = local_weight
        self.global_weight = 1.0 - local_weight
        self.degree_penalty_factor = degree_penalty_factor

        self.nlp = spacy.load("en_core_web_lg")
        self._vector_cache = {}
        print("Advanced Detector Initialized.")
        print(f" - Global Threshold: {self.threshold}")
        print(f" - Local Coherence Weight: {self.local_weight}")
        print(f" - Hub Node Penalty Factor: {self.degree_penalty_factor}")


    def _get_vector(self, text):
        """Caches spaCy vectors and ensures they are CPU-based NumPy arrays."""
        if text not in self._vector_cache:
            doc = self.nlp(text)
            vec = doc.vector if doc.has_vector else np.zeros(self.nlp.vocab.vectors_length)
            self._vector_cache[text] = vec.get() if hasattr(vec, 'get') else vec
        return self._vector_cache[text]

    def _calculate_similarity(self, vec1, vec2):
        """Calculates cosine similarity for NumPy arrays."""
        if np.all(vec1 == 0) or np.all(vec2 == 0): return 0.0
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def find_sentence_cluster(self, starting_nodes):
        """
        Performs an advanced traversal to find the sentence cluster.
        """
        if not starting_nodes:
            return {}, []

        # 1. Create the robust, weighted semantic anchor
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

        # 2. Setup for traversal
        sentence_cluster = {}
        crossed_boundary_edges = []
        queue = list(starting_nodes)
        visited = set(starting_nodes)

        for node in starting_nodes:
            coherence = self._calculate_similarity(self._get_vector(node), anchor_vector)
            sentence_cluster[node] = max(0, 1.0 - coherence)

        # 3. Perform the intelligent traversal
        while queue:
            current_node = queue.pop(0)
            
            # Explore in both directions
            # For each neighbor, we also need the edge data (the verb)
            edges_to_explore = list(self.graph.out_edges(current_node, data=True)) + \
                               list(self.graph.in_edges(current_node, data=True))

            for u, v, edge_data in edges_to_explore:
                # Determine the neighbor node (it's 'v' for out-edges, 'u' for in-edges)
                neighbor_node = v if u == current_node else u

                if neighbor_node not in visited:
                    visited.add(neighbor_node)
                    
                    # --- THE ADVANCED DECISION LOGIC ---

                    # a) Get vectors for all parts of the triplet (Subj, Verb, Obj)
                    current_vec = self._get_vector(current_node)
                    neighbor_vec = self._get_vector(neighbor_node)
                    verb_vec = self._get_vector(edge_data.get('verb', ''))

                    # b) Calculate the different coherence scores
                    global_coherence = self._calculate_similarity(neighbor_vec, anchor_vector)
                    local_coherence = self._calculate_similarity(neighbor_vec, current_vec)
                    verb_coherence = self._calculate_similarity(verb_vec, anchor_vector)
                    
                    # c) Blend the scores
                    blended_score = (self.global_weight * global_coherence) + \
                                    (self.local_weight * local_coherence) + \
                                    (0.1 * verb_coherence) # Small bonus for verb coherence

                    # d) Apply the structural penalty for hub nodes
                    degree = self.graph.degree(neighbor_node)
                    penalty = 1.0 - (np.log1p(degree) / self.degree_penalty_factor)
                    final_score = blended_score * penalty
                    
                    # --- THE FINAL DECISION ---
                    if final_score >= self.threshold:
                        # Coherent: Continue traversal
                        entropy = max(0, 1.0 - final_score)
                        sentence_cluster[neighbor_node] = entropy
                        queue.append(neighbor_node)
                    else:
                        # Boundary Detected: Halt traversal
                        crossed_boundary_edges.append({
                            "from": current_node,
                            "to": neighbor_node,
                            "final_score": final_score
                        })
                        
        return sentence_cluster, crossed_boundary_edges