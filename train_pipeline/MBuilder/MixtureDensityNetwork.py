"""
Contains PyTorch model code for training and evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class MixtureDensityNetwork(nn.Module):

    num_gaussians = None
    output_dim = None
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MixtureDensityNetwork, self).__init__()
        MixtureDensityNetwork.num_gaussians = num_gaussians
        MixtureDensityNetwork.output_dim = output_dim
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(500, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, output_dim * num_gaussians)
        self.sigma = nn.Linear(64, num_gaussians)
        self.pi = nn.Linear(64, num_gaussians)
    
    def forward(self, x):
        x = self.shared_layers(x)
        # Mu
        mus = self.mu(x)
        # Sigma
        # Sigma should be positive and we want values far from 0
        # since in distributions close to 0 can cause numerical instability. Hence we use a modified
        # ELU activation function
        sigmas = nn.ELU()(self.sigma(x)) + 1 + 1e-6
        # sigmas = nn.Softplus()(self.sigma(x)) + 1e-6
        # Pi
        #pis = F.gumbel_softmax(self.pi(x), hard=True)
        pis = F.softmax(self.pi(x), dim=1)

        # Concatenate the outputs
        res = torch.cat([mus, sigmas, pis], dim=1)

        return res
    

    
    def predict(self, dataloader, device):
        """
        Predict from a dataloader and yield the results.
        """
        y_true_final = torch.tensor([]).to(device)
        y_pred_final = torch.tensor([]).to(device)
        # Set the model to evaluation
        self.eval()
        # Turn off gradients
        with torch.no_grad():
            pbar = tqdm(iterable=enumerate(dataloader),
                        total=len(dataloader),
                        desc="Predicting",
                        unit="batch")
            for i, (features, labels) in pbar:
                # Move the data to the device
                features = features.to(device)
                labels = labels.to(device)

                # Make predictions
                parameters = self(features)
                
                # Separate the parameters
                comp = parameters.view(-1, 10 + 2, 50)
                mu_pred = comp[:, :10, :]
                sigma_pred = comp[:, 10, :]
                alpha_pred = comp[:, 10 + 1, :] 
                alpha_pred = alpha_pred.cpu().numpy()
                dim = alpha_pred.shape[1]
                y_pred = np.zeros((len(mu_pred)))  
                y_pred = np.array([mu_pred[i,:,np.random.choice(dim,p=alpha_pred[i])] 
                    for i in np.arange(len(alpha_pred))])  
                y_pred = torch.tensor(y_pred).to(device)

                # True labels
                y_true_final = torch.cat([y_true_final, labels], dim=0)

                # Predicted labels
                y_pred_final = torch.cat([y_pred_final, y_pred], dim=0)

        return y_true_final, y_pred_final


def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max, _ = torch.max(x, dim=axis, keepdim=True)
    return torch.log(torch.sum(torch.exp(x - x_max), dim=axis, keepdim=True)) + x_max


def mean_log_Gaussian_like(parameters, y_true, c=10, m=50):
    """Mean Log Gaussian Likelihood distribution"""
    batch_size = y_true.size(0)

    components = parameters.view(-1, c + 2, m)
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = F.softmax(alpha, dim=1).clamp(1e-8, 1.0)

    pi_term = torch.tensor(2 * 3.141592653589793, dtype=torch.float32, device=parameters.device)

    exponent = torch.log(alpha) - 0.5 * float(c) * torch.log(pi_term) \
        - float(c) * torch.log(sigma) \
        - torch.sum((y_true.unsqueeze(2) - mu)**2, dim=1) / (2 * (sigma)**2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = -torch.mean(log_gauss)
    return res
