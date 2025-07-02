# kg_builder.py (Final Version with Correct SVO Extraction)

import spacy
import networkx as nx
from collections import defaultdict
import pickle
from tqdm import tqdm
import argparse
import os

from utils import download_gutenberg_text

# --- GPU Acceleration (for Colab/NVIDIA GPUs) ---
try:
    spacy.require_gpu()
    print("GPU requirement set. spaCy will use the GPU if available.")
except:
    print("GPU not available or 'cupy' not installed. Running on CPU.")


class SVO_KG_Builder:
    """
    Builds a Knowledge Graph from clean SVO triplets.
    Fixes the issue of nodes being entire sentence fragments.
    """
    def __init__(self, model_name="en_core_web_lg"):
        print(f"Loading spaCy model '{model_name}'...")
        self.nlp = spacy.load(model_name)
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer', first=True)
        self.nlp.max_length = 3500000 
        print("spaCy model loaded and sentencizer is configured.")

    def _get_clean_phrase(self, token):
        """
        THE CRITICAL FIX: Extracts a clean noun phrase.
        It includes determiners, adjectives, and compound nouns, but stops
        at conjunctions or other clauses, preventing the "mega-node" problem.
        """
        phrase_tokens = []
        # Add left-hand modifiers (det, amod, compound, poss)
        for left in token.lefts:
            if left.dep_ in ('det', 'amod', 'compound', 'poss'):
                phrase_tokens.extend(list(left.subtree)) # Add the modifier and its own compounds
        
        # Add the head token itself
        phrase_tokens.append(token)
        
        # Add right-hand modifiers (less common for simple phrases but good to have)
        for right in token.rights:
            if right.dep_ in ('amod', 'compound'):
                phrase_tokens.extend(list(right.subtree))

        # Remove duplicates and sort by original document order
        phrase_tokens = sorted(list(set(phrase_tokens)), key=lambda t: t.i)
        
        # Handle cases like "he and his brother"
        # We only want the part before the conjunction
        final_tokens = []
        for t in phrase_tokens:
            if t.dep_ == 'cc': # 'cc' is a coordinating conjunction (e.g., 'and', 'or')
                break
            final_tokens.append(t)

        return " ".join(t.text for t in final_tokens).strip()


    def build(self, text, min_sent_len=5):
        graph = nx.DiGraph()
        sentence_to_nodes = defaultdict(set)

        print("Step 1: Splitting the book into individual sentences...")
        disabled_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "sentencizer"]
        initial_doc = self.nlp(text, disable=disabled_pipes)
        sentences = [sent.text for sent in initial_doc.sents if len(sent) >= min_sent_len]
        
        print(f"Step 2: Processing {len(sentences)} sentences on the GPU/CPU...")
        
        sent_id = 0
        
        # Use n_process=1 for GPU, or -1 for multi-core CPU
        # Adjust batch_size based on your RAM/VRAM
        process_count = 1 if spacy.prefer_gpu() else -1
        batch_size = 128 if spacy.prefer_gpu() else 50
        
        for doc in tqdm(self.nlp.pipe(sentences, n_process=process_count, batch_size=batch_size), total=len(sentences)):
            for token in doc:
                if token.pos_ == "VERB" or token.pos_ == "AUX":
                    subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr", "xcomp")]

                    if subjects and objects:
                        verb_phrase = self._get_clean_phrase(token)
                        for s_token in subjects:
                            s_phrase = self._get_clean_phrase(s_token)
                            for o_token in objects:
                                o_phrase = self._get_clean_phrase(o_token)
                                # Ensure we don't add empty strings
                                if s_phrase and o_phrase and verb_phrase:
                                    graph.add_node(s_phrase)
                                    graph.add_node(o_phrase)
                                    graph.add_edge(s_phrase, o_phrase, verb=verb_phrase, sentence_id=sent_id)
                                    sentence_to_nodes[sent_id].add(s_phrase)
                                    sentence_to_nodes[sent_id].add(o_phrase)
            sent_id += 1 

        return graph, dict(sentence_to_nodes)

# This part of the script runs when you execute `python kg_builder.py`
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a clean Knowledge Graph from a book.")
    parser.add_argument("--url", type=str, default="https://www.gutenberg.org/ebooks/2600.txt.utf-8", help="URL to the text file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output files.")
    parser.add_argument("--output_file", type=str, default="war_and_peace_kg.pkl", help="Name for the output pickle file.")
    
    args, unknown = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    book_text = download_gutenberg_text(args.url)
    
    kg_builder = SVO_KG_Builder()
    graph, sentence_map = kg_builder.build(book_text)

    print(f"\nKG Build Complete.")
    print(f"  - Nodes: {graph.number_of_nodes()}")
    print(f"  - Edges: {graph.number_of_edges()}")

    print(f"Saving Knowledge Graph and sentence map to '{output_path}'...")
    with open(output_path, "wb") as f:
        pickle.dump({"graph": graph, "sentence_map": sentence_map}, f)
    
    print(f"\nSave complete.")
    print("You can now run your GNN trainer script on this new, clean graph.")