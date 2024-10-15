import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer

class PGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: initial and final temperature tuple. The schedule starts at inital and gradually gets to final as epochs increase
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """
    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.003, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0),sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias

        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2 #since we are concatenating two nodes together to create edge embedding put into MLP
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3


    def _create_explainer_input(self, edge_index, embeds, node_id):
        """
        Given the embedding of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """

        source_node_labels = edge_index[0] #pair is the edge index
        target_node_labels = edge_index[1]

        source_node_embeddings = embeds[source_node_labels]
        target_node_embeddings = embeds[target_node_labels]
        
        edge_embeddings = torch.cat([source_node_embeddings, target_node_embeddings], 1) #concatenated embedding between all pairs of nodes
        return edge_embeddings


    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make sampling less deterministic
        :param training: If set to false, the sampling will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)

            #Gumbel softmax
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature

            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph


    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        #Punishing explainer for weighing too many edges as important
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred) #binary cross entropy with logits

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, indices=range(0,800)):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model (the MLP) is done here to make sure that the seed is set; this spits out the latent variables
        self.explainer_model = nn.Sequential(nn.Linear(self.expl_embedding, 64), nn.ReLU(), nn.Linear(64, 1))

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))

        all_losses_per_epoch = self.train(indices=indices)
        
        return np.array(all_losses_per_epoch)


    def train(self, indices=None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        
        # Make sure the explainer model can be trained
        self.explainer_model.train()
        
        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        
        # as epochs increase the temperature decreases
        def temp_schedule(epoch):
            temp_start, temp_end = self.temp
            return temp_start * ((temp_end / temp_start) ** (epoch / self.epochs))
    
        all_losses = []

        # Start training loop
        for epoch in range(0, self.epochs):
            optimizer.zero_grad()
            epoch_loss = torch.FloatTensor([0]).detach() #epoch loss tensor = 0 

            #initial and final temperature tuple; the schedule starts at inital and gradually gets to final as epochs increase
            t = temp_schedule(epoch)

            #for each epoch do this for all the graphs; this is full batch GD
            for n in indices:
                node_features = self.features[n].detach() #node features all 1
                graph = self.graphs[n].detach() #graph refers to edge index of graph
                node_embeddings = self.model_to_explain.embedding(node_features, graph).detach() #Message passing for node embeddings
                
                #Edge embeddings, input to MLP
                edge_embeddings = self._create_explainer_input(graph, node_embeddings, n).unsqueeze(0)
                
                #Input into MLP
                sampling_weights = self.explainer_model(edge_embeddings)
                
                #Account for temperature schedule to get edge importance
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

                #Get masked and original output MLP layers
                masked_pred = self.model_to_explain(node_features, graph, edge_weights=mask)
                orig_output_layer = self.model_to_explain(node_features, graph)

                #1 or 0 for house/cycle
                original_pred = torch.argmax(orig_output_layer).unsqueeze(0)
                
                #Calculate loss
                id_loss = self._loss(masked_pred, original_pred, mask, self.reg_coefs)
                epoch_loss += id_loss

            if ((epoch + 1) % 5 == 0):
                print(f"Epoch {epoch+1} Total Loss: {epoch_loss.item() :.4f}")
            
            all_losses.append(epoch_loss.item())
            
            #GD and Backprop until our MLP gets good at outputting edge importance
            epoch_loss.backward()
            optimizer.step()
        
        print("\nTraining completed.")
        return all_losses

    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. 
        :param index: index of the node/graph that we wish to explain
        :return: explanation edge index graph and edge weights
        """
        index = int(index)
        
        features = self.features[index].clone().detach()
        graph = self.graphs[index].clone().detach()
        node_embeddings = self.model_to_explain.embedding(features, graph).detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, node_embeddings, index).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze() #deterministic sampling (output is always the same given same input)


        return graph, features, mask #graph = edge index
