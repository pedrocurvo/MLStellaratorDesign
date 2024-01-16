"""
Contains PyTorch model code for training and evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),


            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class MixedNetwork(nn.Module):
    def __init__(self):
        super(MixedNetwork, self).__init__()
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Regression head (9 outputs for regression)
        self.regression_head = nn.Linear(64, 9)
        # Classification head (10 outputs for classification)
        self.classification_head = nn.Linear(64, 10)

    def forward(self, x):
        # Shared layers
        x = self.shared_layers(x)
        # Regression output
        regression_output = self.regression_head(x)
        # Classification output
        classification_output = self.classification_head(x)
        classification_output = F.log_softmax(classification_output, dim=1)

        # Combining the outputs
        combined_output = torch.cat((regression_output[:, :6], 
                                     classification_output, 
                                     regression_output[:, 6:]), dim=1)
        return combined_output