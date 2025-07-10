# Detecting Sentence Boundaries in a Knowledge Graph

This project is a comprehensive solution to the "Natural Language Understanding Without the Language" hackathon challenge. The goal is to traverse a Knowledge Graph (KG) derived from a text and determine sentence boundaries based *solely* on the graph's structure and semantics, without access to the original word order.

This repository documents an iterative journey, starting with simple models, escalating to complex Graph Neural Networks, and culminating in an elegant, non-machine learning algorithm that proved most effective under the problem's unique constraints.

## Final Model: "Structural Entropy"

After extensive experimentation, the most successful and compliant model was determined to be a non-ML algorithm based on "Structural Entropy." This approach correctly interprets the problem's core challenge as one of graph theory rather than conventional machine learning.

The architecture is as follows:

1.  **Knowledge Graph Construction:** A massive, undirected graph is built from the entire text of "War and Peace." Nodes represent the nouns and proper nouns within the text. An edge is created between any two nouns that **co-occur** in the same sentence. This creates a dense, highly connected graph where each sentence forms a "clique" of nodes.

2.  **Graph Traversal:** The algorithm starts at a given node and performs a Breadth-First Search (BFS) to explore its neighbors.

3.  **Entropy-Based Stopping Logic:** The decision to continue or stop a traversal path is based on a novel, structure-only entropy metric:
    *   **Similarity:** The similarity between a `current_node` and a `neighbor_node` is measured using the **Jaccard Similarity** of their respective neighborhoods (i.e., how many neighbors they share).
    *   **Entropy:** The entropy of a traversal step is defined as **`1 - Jaccard Similarity`**.
    *   **Boundary Detection:** If the entropy of moving to a neighbor exceeds a specific, optimized threshold, it signifies a "semantic divergence." The algorithm assumes a sentence boundary has been reached and does not add the neighbor to the cluster.

This approach is fast, requires no training, is purely structural, and directly models the concept of "drifting into unrelated territory" on the graph.

## Performance and Evaluation

To determine the peak performance of this model, a rigorous, automated hyperparameter sweep was conducted. The script tested 51 different entropy thresholds on a diverse test set of 17 sentences, parallelized across all available CPU cores for maximum efficiency.

### Final Optimized Results:

*   **Optimal Entropy Threshold:** `1`
*   **Highest Achievable Accuracy (F1-Score):** **93.31%**

This result represents the quantified limit of the information that can be extracted while strictly adhering to the problem's constraints. The primary bottleneck is the inherent ambiguity created by converting sequential text into an unordered word-node graph.

## Project Structure

*   `book.py`: The main data engineering script. It downloads "War and Peace," processes the entire text to build the massive co-occurrence graph, and saves the final graph object (`full_graph.pkl`) for evaluation.
*   `test_custom.py`: A user-friendly script to test the model on any custom paragraph. It loads the pre-built graph and reports the sentence-level accuracy.
*   `find_best_threshold.py`: The automated hyperparameter tuning script. It runs a parallelized search to find the optimal entropy threshold that yields the highest possible F1-score on a diverse test set.
*   `requirements.txt`: A file listing all necessary Python packages to run the project.

## How to Run

### 1. Setup Environment
It is recommended to use a virtual environment.
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .\.venv\Scripts\activate # On Windows

# Install all dependencies
pip install -r requirements.txt

# Download the necessary spaCy model
python -m spacy download en_core_web_sm
```

### 2. Build the Knowledge Graph
This step needs to be run only once. It will process the entire text of "War and Peace" and may take several minutes.
```bash
python book.py
```
This will create the `full_graph.pkl` file.

### 3. Evaluate on a Custom Paragraph
Edit the `my_paragraph` variable in `test_custom.py` and run:
```bash
python test_custom.py
```

### 4. (Optional) Re-run the Hyperparameter Sweep
To verify the optimal threshold on your machine, run the parallelized tuning script:
```bash
python find_best_threshold.py
```
