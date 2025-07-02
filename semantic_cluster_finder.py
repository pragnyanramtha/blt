# semantic_cluster_finder.py (Final, Enriched Graph Version)

import spacy
import networkx as nx
import numpy as np
from collections import defaultdict
import random

# Ensure you have the spaCy model downloaded:
# python -m spacy download en_core_web_lg

class SVO_KG_Builder:
    """A helper class to build a richer, more connected Knowledge Graph."""
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
        final_tokens = [t for t in phrase_tokens if t.dep_ != 'cc']
        return " ".join(t.text for t in final_tokens).strip()

    def build(self, text):
        """
        Builds a richer graph by extracting both S-V-O and Noun-Prep-Noun triplets.
        """
        doc = self.nlp(text)
        graph = nx.DiGraph()
        sentence_to_nodes = defaultdict(set) # For testing/oracle function
        
        for sent_id, sent in enumerate(doc.sents):
            for token in sent:
                # 1. Extract Subject-Verb-Object triplets
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
                                    graph.add_edge(s_phrase, o_phrase, verb=verb_phrase)
                                    sentence_to_nodes[sent_id].update([s_phrase, o_phrase])

                # 2. THE FIX: Extract Noun-Preposition-Noun relationships to connect phrases
                if token.pos_ in ("NOUN", "PROPN"):
                    for child in token.children:
                        if child.dep_ == 'prep': # Found a preposition
                            prep_text = child.text
                            for p_child in child.children:
                                if p_child.dep_ == 'pobj': # Found a prepositional object
                                    head_phrase = self._get_clean_phrase(token)
                                    obj_phrase = self._get_clean_phrase(p_child)
                                    if head_phrase and obj_phrase:
                                        graph.add_edge(head_phrase, obj_phrase, verb=prep_text)
                                        sentence_to_nodes[sent_id].update([head_phrase, obj_phrase])

        return graph, dict(sentence_to_nodes)

class LiveBoundaryDetector:
    """
    Traverses a KG from starting nodes and detects sentence boundaries
    based on semantic coherence. This is the core model.
    """
    def __init__(self, coherence_threshold=0.70):
        self.threshold = coherence_threshold
        self.nlp = spacy.load("en_core_web_lg")
        self._vector_cache = {}

    def _get_vector(self, text):
        if text not in self._vector_cache:
            self._vector_cache[text] = self.nlp(text).vector
        return self._vector_cache[text]

    def _calculate_similarity(self, vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0): return 0.0
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def find_sentence_cluster(self, graph, starting_nodes):
        """
        The main detection method. Receives a graph and starting nodes,
        with no knowledge of the original text.
        """
        valid_start_nodes = [node for node in starting_nodes if node in graph]
        if not valid_start_nodes:
            print(f"Error: None of the starting nodes {starting_nodes} were found in the generated graph.")
            return {}, []

        start_vectors = [self._get_vector(node) for node in valid_start_nodes]
        anchor_vector = np.mean([v for v in start_vectors if not np.all(v == 0)], axis=0)
        
        sentence_cluster = {}
        crossed_boundary_edges = []
        queue = list(valid_start_nodes)
        visited = set(valid_start_nodes)

        for node in valid_start_nodes:
            coherence = self._calculate_similarity(self._get_vector(node), anchor_vector)
            entropy = max(0, 1.0 - coherence)
            sentence_cluster[node] = entropy

        while queue:
            current_node = queue.pop(0)
            neighbors = set(graph.successors(current_node)) | set(graph.predecessors(current_node))
            
            for neighbor_node in neighbors:
                if neighbor_node in visited: continue
                visited.add(neighbor_node)
                
                coherence_score = self._calculate_similarity(self._get_vector(neighbor_node), anchor_vector)
                if coherence_score >= self.threshold:
                    entropy = max(0, 1.0 - coherence_score)
                    sentence_cluster[neighbor_node] = entropy
                    queue.append(neighbor_node)
                else:
                    crossed_boundary_edges.append({"from": current_node, "to": neighbor_node, "coherence": coherence_score})
                        
        return sentence_cluster, crossed_boundary_edges

def get_hidden_graph_data(paragraph, sentence_index_to_test):
    """
    THE ORACLE: Prepares data, but only returns what the model is allowed to see.
    """
    print(f"Oracle: Building enriched graph from a hidden paragraph...")
    kg_builder = SVO_KG_Builder()
    full_graph, sentence_map = kg_builder.build(paragraph)
    
    if sentence_index_to_test not in sentence_map:
        print(f"Error: Sentence index {sentence_index_to_test} not found.")
        return None, None
        
    start_nodes = list(sentence_map[sentence_index_to_test])
    
    print("Oracle: Providing the full, connected graph and a list of starting nodes.")
    print("-" * 20)
    return full_graph, start_nodes


if __name__ == "__main__":
    example_paragraph = (
        "The dreadful news of the battle of Borodinó, of our losses in killed and wounded, and the still more terrible news of the loss of Moscow reached Vorónezh in the middle of September Princess Mary, having learned of her brother’s wound only from the Gazette and having no definite news of him, prepared (so Nicholas heard, he had not seen her again himself) to set off in search of Prince Andrew"
        "The dreadful news of the battle of Borodinó, of our losses in killed and wounded, and the still more terrible news of the loss of Moscow reached Vorónezh in the middle of September Princess Mary, having learned of her brother’s wound only from the Gazette and having no definite news of him, prepared (so Nicholas heard, he had not seen her again himself) to set off in search of Prince Andrew"
        "The dreadful news of the battle of Borodinó, of our losses in killed and wounded, and the still more terrible news of the loss of Moscow reached Vorónezh in the middle of September Princess Mary, having learned of her brother’s wound only from the Gazette and having no definite news of him, prepared (so Nicholas heard, he had not seen her again himself) to set off in search of Prince Andrew"
    )

    print("Simulating a scenario where paragraph and sentence boundaries are hidden from the model...")
    
    hidden_graph, start_nodes = get_hidden_graph_data(example_paragraph, sentence_index_to_test=1)

    if hidden_graph and start_nodes:
        print(f"Model receives a graph with {hidden_graph.number_of_nodes()} nodes and {hidden_graph.number_of_edges()} edges.")
        print(f"Model receives starting nodes: {start_nodes}")
        print("-" * 20)
        
        detector = LiveBoundaryDetector(coherence_threshold=0.68)
        detected_cluster, boundary_crossings = detector.find_sentence_cluster(hidden_graph, start_nodes)
        
        sorted_cluster = sorted(detected_cluster.items(), key=lambda item: item[1], reverse=True)
        
        print("\n--- Model Output ---")
        if not sorted_cluster:
            print("Model did not find a coherent sentence cluster.")
        else:
            print("\n[+] Detected Sentence Cluster (Nodes sorted by entropy):")
            for node, entropy in sorted_cluster:
                print(f"  - Entropy: {entropy:.4f} | Node: '{node}'")
            
            if boundary_crossings:
                print("\n[+] Traversal Halted at Boundaries:")
                for edge in boundary_crossings:
                    print(f"  - From '{edge['from']}' to '{edge['to']}' (Coherence: {edge['coherence']:.2f} was below threshold)")