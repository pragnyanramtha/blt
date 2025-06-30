# kg_builder.py (CORRECTED SVO EXTRACTION)
import spacy
import networkx as nx
from collections import defaultdict
import pickle
from tqdm import tqdm
import argparse, os

# Your other files...
from utils import download_gutenberg_text

class SVO_KG_Builder:
    def __init__(self, model_name="en_core_web_lg"):
        self.nlp = spacy.load(model_name)
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer', first=True)
        self.nlp.max_length = 3500000 

    # THE CRITICAL FIX IS HERE
    def _get_full_phrase(self, token):
        # This is a more robust way to get a clean noun phrase instead of the whole subtree.
        # It includes determiners, adjectives, and compound nouns.
        phrase_tokens = [t for t in token.lefts if t.dep_ in ('det', 'amod', 'compound', 'poss')]
        phrase_tokens.append(token)
        # It's less common to have modifiers on the right for a simple phrase, but can be added.
        phrase_tokens.sort(key=lambda t: t.i)
        return " ".join(t.text for t in phrase_tokens).strip()

    def build(self, text, min_sent_len=5):
        # ... (The rest of the build method is the same as the last working version) ...
        # ... (I'm omitting it here for brevity, but use the full code from our last successful run) ...
        graph = nx.DiGraph()
        sentence_to_nodes = defaultdict(set)
        # ... the rest of the build logic ...
        return graph, dict(sentence_to_nodes)

# (Make sure the full build method and the __main__ block are here from the previous version)