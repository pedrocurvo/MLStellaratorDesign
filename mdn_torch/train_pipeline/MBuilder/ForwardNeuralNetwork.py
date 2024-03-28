"""
Contains PyTorch model for forward neural network.
"""
import torch.nn as nn
import torch.nn.functional as F

class ForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ForwardNeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x
    