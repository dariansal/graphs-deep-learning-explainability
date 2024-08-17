#BA2MOTIFS and Custom BA2MOTIFS
import torch
from torch_geometric.datasets import BA2MotifDataset, ExplainerDataset
from torch_geometric.data import Data, Dataset
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected, degree, to_networkx, from_networkx
import matplotlib.pyplot as plt

#Custom BA2MOTIFS
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif, CycleMotif

#BA2MOTIFS graph from scratch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx #focuses on network analysis and graph theory; can create/visualize graphs
import inspect #for viewing source code
import pprint
from tqdm import tqdm
import pickle
import random
from dataclasses import dataclass

#Classifier
import torch.nn as nn
from torch.nn import Linear, Sequential
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_max_pool, global_mean_pool

#Fine tuning
import copy

class GIN(torch.nn.Module):
    def __init__(self, num_features, dropout_rate = 0.2, num_classes = 2, pretrained_model = False):
        super(GIN, self).__init__()
        self.embedding_size = 16

        self.nn1 = Sequential(
            Linear(num_features, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            Linear(32, 16),
            nn.ReLU())

        self.conv1 = GINConv(self.nn1)
        self.bn1 = nn.BatchNorm1d(16)

        self.nn2 = Sequential(
            Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            Linear(16, 16),
            nn.ReLU())
        
        self.conv2 = GINConv(self.nn2)
        self.bn2 = nn.BatchNorm1d(16)

        self.fc1 = Linear(32, num_classes) #2 classes

        #Ensure original model isn't altered during fine-tuning
        if pretrained_model:
            temp = torch.load(f'../models/BA2-Scratch/{pretrained_model}')
            cloned_state_dict = copy.deepcopy(temp)
            self.load_state_dict(cloned_state_dict)
            
            
            self.freeze_layers()
            self.fc1 = Linear(32, 2) #Common to freeze earlier layers; replace desired layer for fine tuning

    def freeze_layers(self):
        for name, param in self.named_parameters():
                param.requires_grad = True if 'fc1' else False # Don't freeze the final layer

    def embedding(self, x, edge_index, edge_weights = None):
        # Process through GIN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return x
    
    def forward(self, x, edge_index, batch = None, edge_weights = None):
        x, edge_index = x, edge_index #extracts feature matrix from graph, edge info, batch indices
        node_embeddings = self.embedding(x, edge_index) #reduces dimension of nodes and edges

        x_max = global_max_pool(node_embeddings, batch)
        x_mean = global_mean_pool(node_embeddings, batch)
        x = torch.cat([x_max, x_mean], dim=1) #combines these two poolings into single tensor

        return self.fc1(x)