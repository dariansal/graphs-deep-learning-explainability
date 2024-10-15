import networkx as nx
import torch
import matplotlib.pyplot as plt


def highlight_explanation(edge_index, edge_weights, num_top_edges, show=True):
    """
    Function that plots the entire explanation graph with important edges highlighted.
    :param graph: edge_index provided by explainer
    :param edge_weights: Mask of edge weights provided by explainer
    :param num_top_edges: number of top edges to highlight
    :param show: flag to show plot made
    """
    # Set threshold for important edges
    sorted_edge_weights, _ = torch.sort(edge_weights, descending=True)
    num_edges = edge_weights.shape[0]
    thres = sorted_edge_weights[min(num_top_edges, num_edges) - 1] #10 edges, want top 4, threshold at index 3 for descending order of weights

    # Initialize graph object
    G = nx.Graph()

    # Add all edges to the graph
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    for i in range(num_edges):
        if source_nodes[i] != target_nodes[i]:  # Exclude self-loops
            G.add_edge(source_nodes[i].item(), target_nodes[i].item(), importance=edge_weights[i].item()) #adds edge attribute of weight/importance; order preserved for edge_weights

    ####HIGHLIGHTED EDGES ARE all the edges that pass threshold importance and are in largest important subgraph

    important_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['importance'] >= thres]
    subG = nx.Graph(important_edges) #creates subgraph of only the important edges and associated nodes; some parts could be disconnected from others

    '''Find the largest connected component in the subgraph. nx.connected_components(subG) is an iterator of sets, where each set contains nodes
    of one connected component in subG. An example of this is [{1, 2, 3}, {4, 5}, {6, 7, 8, 9}]. max() iterates through all of these, calls the key 
    function on all of them, and returns the "largest" one based on the key of length, so the component with the most nodes.largest_cc, a list of nodes, is returned'''

    largest_cc = max(nx.connected_components(subG), key=len) 
    highlight_edges = [(u, v) for (u, v) in important_edges if u in largest_cc and v in largest_cc] #should be the motif edges
    
    # The rest of the edges
    other_edges = [(u, v) for (u, v) in G.edges() if (u, v) not in highlight_edges]

    return highlight_edges, other_edges