# boundary_detector.py
import spacy
import numpy as np

class BoundaryDetector:
    def __init__(self, graph, coherence_threshold=0.6):
        self.graph = graph
        self.threshold = coherence_threshold
        self.nlp = spacy.load("en_core_web_lg")
        self._vector_cache = {}

    def _get_vector(self, text):
        if text not in self._vector_cache:
            doc = self.nlp(text)
            vec = doc.vector if doc.has_vector else np.zeros(self.nlp.vocab.vectors_length)
            if hasattr(vec, 'get'):
                self._vector_cache[text] = vec.get()
            else:
                self._vector_cache[text] = vec
        return self._vector_cache[text]

    def _calculate_similarity(self, vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0): return 0.0
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def find_sentence_cluster(self, starting_nodes):
        if not starting_nodes: return {}, []

        weighted_vectors, total_weight = [], 0
        print("  [Model] Creating weighted semantic anchor...")
        for node_text in starting_nodes:
            node_doc = self.nlp(node_text, disable=["parser", "ner"])
            first_token_pos = node_doc[0].pos_ if len(node_doc) > 0 else "NOUN"
            weight = 3.0 if first_token_pos in ["PROPN", "NOUN"] else 0.5 if first_token_pos == "PRON" else 1.0
            print(f"    - Node: '{node_text}', POS: {first_token_pos}, Weight: {weight}")
            vec = self._get_vector(node_text)
            if not np.all(vec == 0):
                weighted_vectors.append(vec * weight)
                total_weight += weight

        if not weighted_vectors: return {}, []
        anchor_vector = np.sum(weighted_vectors, axis=0) / total_weight

        sentence_cluster, crossed_boundary_edges, queue, visited = {}, [], list(starting_nodes), set(starting_nodes)
        for node in starting_nodes:
            coherence = self._calculate_similarity(self._get_vector(node), anchor_vector)
            sentence_cluster[node] = max(0, 1.0 - coherence)

        while queue:
            current_node = queue.pop(0)
            neighbors = set(self.graph.successors(current_node)) | set(self.graph.predecessors(current_node))
            for neighbor_node in neighbors:
                if neighbor_node not in visited:
                    visited.add(neighbor_node)
                    coherence_score = self._calculate_similarity(self._get_vector(neighbor_node), anchor_vector)
                    if coherence_score >= self.threshold:
                        sentence_cluster[neighbor_node] = max(0, 1.0 - coherence_score)
                        queue.append(neighbor_node)
                    else:
                        crossed_boundary_edges.append({"from": current_node, "to": neighbor_node, "coherence": coherence_score})
        return sentence_cluster, crossed_boundary_edges