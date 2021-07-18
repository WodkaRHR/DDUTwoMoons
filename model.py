import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class ResFFN(nn.Module):
    """ Feed Forward Network with Residual Connections and Spectral Normalization. """

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4, leaky_relu_slope=0.01, relu_in_last_hidden=True):
        """ Initializes the FFN. 

        Parameters:
        -----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of classes
        hidden_dim : int
            The hidden dimensionality
        num_layers : int
            How many hidden layers are present.
        leaky_relu_slope : float
            Slope for negative activations in the leaky relu.
        relu_in_last_hidden : bool
            If False, no non-linearity will be applied after the last hidden layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leaky_relu_slope = leaky_relu_slope
        self.relu_in_last_hidden = relu_in_last_hidden

        self.first_linear = nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim))
        self.layers = nn.ModuleList(
            [nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)]
        )
        self.last_linear = nn.utils.spectral_norm(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, return_features=False):
        """ Forward pass through the FNN. 
        
        Parameters:
        -----------
        x : torch.Tensor, shape [batch_size, input_dim]
            Inputs to the network.
        return_features : bool
            If True, the model serves as feature extractor and doesn't project into output_dim-dimensional logits.
        
        Returns:
        --------
        logits : torch.Tensor, shape [batch_size, num_classes]
            Logits for each class.
            If 'return_features' is True, instead it returns a tensor of shape [batch_size, hidden_dims[-1]] as feature representations of the input.
        """
        x = self.first_linear(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        res = x
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1 or self.relu_in_last_hidden: # Don't use a non-linearity in the last hidden layer to prevent singular covariance matrices
                x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
            x += res
            res = x
        if not return_features:
            x = self.last_linear(x)
        return x
    