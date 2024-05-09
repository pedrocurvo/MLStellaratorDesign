"""
Contains PyTorch model code for Mixture Density Network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class MDNFullCovariance(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MDNFullCovariance, self).__init__()
        self.num_param = int((output_dim * output_dim + 3 * output_dim + 2) / 2)
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
        )

        self.mu = nn.Linear(2048, output_dim * num_gaussians)
        self.sigma_not_in_diagonal = nn.Linear(2048, int(num_gaussians * (output_dim * (output_dim-1)) / 2))
        self.sigma_diag = nn.Linear(2048, num_gaussians * output_dim)
        self.pi = nn.Linear(2048, num_gaussians)
    
    def forward(self, x):
        """
        Forward pass of the model.
        """
        # Pass through the shared layers
        x = self.shared_layers(x)
        # Mus (means of the gaussians)
        mus = self.mu(x)
        # Sigmas (covariance matrices of the gaussians)
        sigmas_not_in_diagonal = self.sigma_not_in_diagonal(x)
        sigmas_diag = self.sigma_diag(x)
        # Pis (mixture weights, softmaxed since they need to sum to 1)
        pis = F.softmax(self.pi(x) - self.pi(x).max(), dim=1)

        # Concatenate the outputs
        res = torch.cat([mus, sigmas_not_in_diagonal, sigmas_diag, pis], dim=1)

        return res
    
    
    # ------------------------------------------------------------------------------------------------------------------
    def log_prob_loss(self, parameters, y_true):
        # Un-cat the parameters
        mu = parameters[:, :self.output_dim * self.num_gaussians]
        sigma_not_in_diagonal = parameters[:, self.output_dim * self.num_gaussians:self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2)]
        sigma_diag = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2):self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim]
        alpha = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim:]

        # Reshape the parameters to have the correct dimensions
        # Example: mu: (batch_size, output_dim, num_gaussians)
        mu = mu.view(-1,  self.num_gaussians, self.output_dim)
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(-1, self.num_gaussians, int((self.output_dim * (self.output_dim-1)) / 2))
        sigma_diag = sigma_diag.view(-1, self.num_gaussians, self.output_dim)
        alpha = alpha.view(-1, self.num_gaussians)

        # Tensor to store L (lower triangular matrix), filled with zeros
        L = torch.zeros((sigma_diag.shape[0], self.num_gaussians, self.output_dim, self.output_dim),
                        device=parameters.device,
                        dtype=parameters.dtype)

        # Compute the indices for the lower triangular portion
        indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=-1,
                                     device=parameters.device)

        # Reshape sigma_not_in_diagonal to have the correct dimensions
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(sigma_not_in_diagonal.shape[0], self.num_gaussians, -1)

        # Fill the lower triangular matrix
        L[:, :, indices[0], indices[1]] = sigma_not_in_diagonal

        # Add the diagonal to the lower triangular matrix, but first pass it through an activation function
        L[:, :, torch.arange(self.output_dim), torch.arange(self.output_dim)] = nn.ELU()(sigma_diag) + 1 + 1e-6

        # Pass sigma_not_in_diagonal and sigma_diag through torch.dist to create a MultivariateNormal distribution
        mvn = dist.MultivariateNormal(mu, scale_tril=L)

        # Use alphas to create a mixture of distributions
        mix = dist.Categorical(alpha)
        mixture = dist.MixtureSameFamily(mix, mvn)

        # Calculate the log likelihood
        log_likelihood = mixture.log_prob(y_true)

        # Calculate the mean log likelihood
        return -torch.mean(log_likelihood)

    # ------------------------------------------------------------------------------------------------------------------
    def getMixturesSample(self, features, device):
        # Move the data to the device
        features = features.to(device)

        # Make predictions
        parameters = self(features)
        
        # Separate the parameters
        # Un-cat the parameters
        mu = parameters[:, :self.output_dim * self.num_gaussians]
        sigma_not_in_diagonal = parameters[:, self.output_dim * self.num_gaussians:self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2)]
        sigma_diag = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2):self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim]
        alpha = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim:]

        # Reshape the parameters to have the correct dimensions
        # Example: mu: (batch_size, output_dim, num_gaussians)
        mu = mu.view(-1,  self.num_gaussians, self.output_dim)
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(-1, self.num_gaussians, int((self.output_dim * (self.output_dim-1)) / 2))
        sigma_diag = sigma_diag.view(-1, self.num_gaussians, self.output_dim)
        alpha = alpha.view(-1, self.num_gaussians)

        # Tensor to store L (lower triangular matrix), filled with zeros
        L = torch.zeros((sigma_diag.shape[0], self.num_gaussians, self.output_dim, self.output_dim),
                        device=parameters.device,
                        dtype=parameters.dtype)

        # Compute the indices for the lower triangular portion
        indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=-1,
                                    device=parameters.device)

        # Reshape sigma_not_in_diagonal to have the correct dimensions
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(sigma_not_in_diagonal.shape[0], self.num_gaussians, -1)

        # Fill the lower triangular matrix
        L[:, :, indices[0], indices[1]] = sigma_not_in_diagonal

        # Add the diagonal to the lower triangular matrix, but first pass it through an activation function
        L[:, :, torch.arange(self.output_dim), torch.arange(self.output_dim)] = nn.ELU()(sigma_diag) + 1 + 1e-6

        # Pass sigma_not_in_diagonal and sigma_diag through torch.dist to create a MultivariateNormal distribution
        mvn = dist.MultivariateNormal(mu, scale_tril=L)

        # Use alphas to create a mixture of distributions
        mix = dist.Categorical(alpha)
        mixture = dist.MixtureSameFamily(mix, mvn)

        return mixture.sample(sample_shape=torch.Size([1])).squeeze(0)
