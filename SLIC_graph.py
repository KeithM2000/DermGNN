from skimage.segmentation import slic
from skimage.graph import rag_mean_color
import numpy as np

def slic_graph(image, num_nodes, compactness):
    # Step 1: SLIC segmentation
    segments = slic(image, n_segments=num_nodes, compactness=compactness)

    # Step 2: Region adjacency graph
    rag = rag_mean_color(image, segments)

    # Step 3: Map segments labels to continuous indices
    unique_labels = np.unique(segments)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indexed_segments = np.vectorize(label_to_index.get)(segments)

    # Step 4: Build the corrected adjacency matrix
    n_superpixels = len(unique_labels)
    adj_matrix = np.zeros((n_superpixels, n_superpixels), dtype=int)
    for edge in rag.edges:
        idx1 = label_to_index[edge[0]]
        idx2 = label_to_index[edge[1]]
        adj_matrix[idx1, idx2] = 1
        adj_matrix[idx2, idx1] = 1

    return indexed_segments, adj_matrix