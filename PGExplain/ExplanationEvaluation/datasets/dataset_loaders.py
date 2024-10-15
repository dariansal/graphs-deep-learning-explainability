import numpy as np
import os
from numpy.random.mtrand import RandomState
import torch


def load_graph_dataset(shuffle=True):
    """Load a graph dataset and optionally shuffle it.

    :param _dataset: Which dataset to load. Choose from "ba2" or "mutag"
    :param shuffle: Boolean. Wheter to shuffle the loaded dataset.
    :returns: np.array
    """
    
    #Load dataset properly
    #__file__ - the path to current script file
    #os.path.realpath() Converts path to an absolute path
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ba2 = torch.load(dir_path + '/serialized-data/custom-ba2.pt')


    features = []
    labels = []
    edge_indices = [] #edge index format of storing graph structure 
    
    #Separate the info for each graph for compatability
    for data in ba2:
        edge_indices.append(data.edge_index)

        features.append(data.x)  # Assuming x contains node features

        label = data.y.item()
        labels.append(label)  # Assuming y contains the label
    
    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    num_graphs = labels.shape[0]

    indices = np.arange(0, num_graphs)
    if shuffle:
        prng = RandomState(42) #Setting seed
        indices = prng.permutation(indices)

    # Create shuffled data
    features = features[indices].astype('float32')
    labels = labels[indices]

    # Create masks (not used)
    train_indices = np.arange(0, int(num_graphs*0.8))
    val_indices = np.arange(int(num_graphs*0.8), int(num_graphs*0.9))
    test_indices = np.arange(int(num_graphs*0.9), num_graphs)
    train_mask = np.full((num_graphs), False, dtype=bool)
    train_mask[train_indices] = True
    val_mask = np.full((num_graphs), False, dtype=bool)
    val_mask[val_indices] = True
    test_mask = np.full((num_graphs), False, dtype=bool)
    test_mask[test_indices] = True

    return edge_indices, features, labels, train_mask, val_mask, test_mask