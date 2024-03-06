"""
Contains PyTorch model code for Mixture Density Network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.distributions as dist

class MDNFullCovariance(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MDNFullCovariance, self).__init__()
        self.num_parameters = int((output_dim * output_dim + 3 * output_dim + 2) / 2)
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, self.num_parameters),
            nn.Tanh(),
            # nn.Linear(self.num_parameters, self.num_parameters),
            # nn.Tanh(),
            nn.Linear(self.num_parameters, self.num_parameters),
            nn.Tanh()
        )
        self.mu = nn.Linear(self.num_parameters, output_dim * num_gaussians)
        self.sigma_not_in_diagonal = nn.Linear(self.num_parameters, int(num_gaussians * (output_dim * (output_dim-1)) / 2))
        self.sigma_diag = nn.Linear(self.num_parameters, num_gaussians * output_dim)
        self.pi = nn.Linear(self.num_parameters, num_gaussians)
    
    def forward(self, x):
        x = self.shared_layers(x)
        # Mu
        mus = self.mu(x)
        # Sigma
        # Sigma should be positive and we want values far from 0
        # since in distributions close to 0 can cause numerical instability. Hence we use a modified
        # ELU activation function
        #sigmas = nn.ELU(alpha=1.)(self.sigma(x)) + 1 + 1e-8
        sigmas_not_in_diagonal = self.sigma_not_in_diagonal(x) + 1e-6
        sigmas_diag = self.sigma_diag(x) + 1e-6 + 1
        # Pi
        pis = F.softmax(self.pi(x) - self.pi(x).max(), dim=1)

        # Concatenate the outputs
        res = torch.cat([mus, sigmas_not_in_diagonal, sigmas_diag, pis], dim=1)

        return res
    
    @staticmethod
    def log_sum_exp(x, axis=None):
        """Log-sum-exp trick implementation"""
        x_max, _ = torch.max(x, dim=axis, keepdim=True)
        return torch.log(torch.sum(torch.exp(x - x_max), dim=axis, keepdim=True)) + x_max


    def loss3(self, parameters, y_true):
        """Mean Log Gaussian Likelihood distribution"""

        # I receive a tensor of shape (batch_size, (output_dim * output_dim + 3 * output_dim + 2) / 2 * num_gaussians)
        # I need to separate the parameters to have:
        # mu: (batch_size, output_dim * num_gaussians)
        # sigma_not_in_diagonal: (batch_size, (output_dim * (output_dim-1)) / 2 * num_gaussians)
        # sigma_diag: (batch_size, output_dim * num_gaussians)
        # alpha: (batch_size, num_gaussians)

        mu = parameters[:, :self.output_dim * self.num_gaussians]
        sigma_not_in_diagonal = parameters[:, self.output_dim * self.num_gaussians:self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2)]
        sigma_diag = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2):self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim]
        alpha = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim:]

        # Reshape the parameters so I have the correct dimensions
        # Example: mu: (batch_size, output_dim, num_gaussians)
        mu = mu.view(-1,  self.num_gaussians, self.output_dim)
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(-1, self.num_gaussians, int((self.output_dim * (self.output_dim-1)) / 2))
        sigma_diag = sigma_diag.view(-1, self.num_gaussians, self.output_dim)
        alpha = alpha.view(-1, self.num_gaussians)

        

        # Pass the sigma_not_in_diagonal through a function to make it a covariance matrix
        # I will use the Cholesky decomposition
        # I will use the following formula to calculate the covariance matrix
        # sigma = L * L^T
        # where L is a lower triangular matrix
        # I will use the following formula to calculate the lower triangular matrix
        # L = diag(exp(sigma_diag)) + tril(sigma_not_in_diagonal) torch.diag_embed(torch.exp(sigma_diag), dim1=-1, dim2=-2)
        U = torch.zeros((sigma_diag.shape[0], self.num_gaussians, self.output_dim, self.output_dim), device=parameters.device, dtype=parameters.dtype)
        for i in range(self.output_dim):
            U[:, :, i, i] = nn.ELU()(sigma_diag[:, :, i]) + 1e-6 + 1
        
        # Fill the upper triangular part of the matrix
        for i in range(self.output_dim):
            for j in range(i+1, self.output_dim):
                U[:, :, i, j] = sigma_not_in_diagonal[:, :, int(i * (self.output_dim - 1) / 2 + j - (i + 1) * (i + 2) / 2)]


        # Reconstruct Covariance matrix
        E = torch.matmul(U, U.transpose(-1, -2))

        # Calculate the determinant of the covariance matrix
        det_sigma = torch.det(E)
        # Invert the covariance matrix
        inv_sigma = torch.inverse(E)
        # Compute the Manhalanobis distance 
        diff = y_true.unsqueeze(1) - mu
        mahalanobis_sq = torch.sum(diff.unsqueeze(-2) @ inv_sigma @ diff.unsqueeze(-1), dim=-1).squeeze(-1)
        # Compute the log PDF 
        log_pdf_gaussian = -0.5 * (self.output_dim * torch.log(2 * torch.tensor(3.141592653589793)) + torch.log(det_sigma) + mahalanobis_sq)
        # Compute the log sum exp with the log PDF sum trick
        log_sum_exp = MDN_FCOV.log_sum_exp(log_pdf_gaussian, axis=1)
        loss = -torch.mean(log_sum_exp)
        return loss




    def loss2(self, parameters, y_true):
        mu = parameters[:, :self.output_dim * self.num_gaussians]
        sigma_not_in_diagonal = parameters[:, self.output_dim * self.num_gaussians:self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2)]
        sigma_diag = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2):self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim]
        alpha = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim:]

        # Reshape the parameters so I have the correct dimensions
        # Example: mu: (batch_size, output_dim, num_gaussians)
        mu = mu.view(-1,  self.num_gaussians, self.output_dim)
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(-1, self.num_gaussians, int((self.output_dim * (self.output_dim-1)) / 2))
        sigma_diag = sigma_diag.view(-1, self.num_gaussians, self.output_dim)
        alpha = alpha.view(-1, self.num_gaussians)

        # Pass the sigma_not_in_diagonal through a function to make it a covariance matrix
        # I will use the Cholesky decomposition
        # I will use the following formula to calculate the covariance matrix
        # sigma = L * L^T
        # where L is a lower triangular matrix
        # I will use the following formula to calculate the lower triangular matrix
        # L = diag(exp(sigma_diag)) + tril(sigma_not_in_diagonal) torch.diag_embed(torch.exp(sigma_diag), dim1=-1, dim2=-2)
        # U = torch.zeros((sigma_diag.shape[0], self.num_gaussians, self.output_dim, self.output_dim), device=parameters.device, dtype=parameters.dtype)
        # for i in range(self.output_dim):
        #     U[:, :, i, i] = nn.ELU()(sigma_diag[:, :, i]) + 1e-6 + 1
        
        # # Fill the upper triangular part of the matrix
        # for i in range(self.output_dim):
        #     for j in range(i+1, self.output_dim):
        #         U[:, :, i, j] = sigma_not_in_diagonal[:, :, int(i * (self.output_dim - 1) / 2 + j - (i + 1) * (i + 2) / 2)]


        # # Reconstruct Covariance matrix
        # E = torch.matmul(U, U.transpose(-1, -2))

        # # Calculate the determinant of the covariance matrix
        # det = torch.det(E)

        # # Invert the covariance matrix
        # E_inv = torch.inverse(E)

        # # Tensor to store L, filled with zeros
        # lower_triangular = torch.zeros((sigma_diag.shape[0], self.num_gaussians, self.output_dim, self.output_dim), device=parameters.device, dtype=parameters.dtype)

        # # Compute the indices for the lower trinagular portion
        # indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=0, device=parameters.device)
        # # Generate all possible row and column indices
        # rows, cols = torch.meshgrid(torch.arange(self.output_dim), torch.arange(self.output_dim))
        # # Mask the indices to exclude the diagonal and upper triangular portion
        # mask = rows > cols
        # indices = torch.stack((rows[mask], cols[mask]), dim=0)

        # # Reshape sigma_not_in_diagonal to have the correct dimensions
        # sigma_not_in_diagonal = sigma_not_in_diagonal.view(sigma_not_in_diagonal.shape[0], self.num_gaussians, -1)

        # # Fill the lower triangular matrix
        # lower_triangular[:, :, indices[0], indices[1]] = sigma_not_in_diagonal

        # # Add the diagonal to the lower triangular matrix
        # lower_triangular[:, :, torch.arange(self.output_dim), torch.arange(self.output_dim)] = sigma_diag
        # Tensor to store L, filled with zeros
        lower_triangular = torch.zeros((sigma_diag.shape[0], self.num_gaussians, self.output_dim, self.output_dim), device=parameters.device, dtype=parameters.dtype)

        # Compute the indices for the lower triangular portion
        indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=-1, device=parameters.device)

        # Reshape sigma_not_in_diagonal to have the correct dimensions
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(sigma_not_in_diagonal.shape[0], self.num_gaussians, -1)

        # Fill the lower triangular matrix
        lower_triangular[:, :, indices[0], indices[1]] = sigma_not_in_diagonal

        # Add the diagonal to the lower triangular matrix
        lower_triangular[:, :, torch.arange(self.output_dim), torch.arange(self.output_dim)] = nn.ELU()(sigma_diag) + 1
        

        # E
        E = torch.matmul(lower_triangular, lower_triangular.transpose(-1, -2))
        E_inv = torch.inverse(E)
        det = torch.det(E)


        # Calculate the log likelihood
        pi_term = torch.tensor(2 * 3.141592653589793, dtype=torch.float32, device=parameters.device)

        # Compute the Manhalanobis distance 
        diff = y_true.unsqueeze(1) - mu
        mahalanobis_sq = torch.sum(diff.unsqueeze(-2) @ E_inv @ diff.unsqueeze(-1), dim=-1).squeeze(-1)
        mahalanobis_sq = mahalanobis_sq

        # v1 = (y_true.unsqueeze(1) - mu).unsqueeze(-2)
        # v2 = (y_true.unsqueeze(1) - mu).unsqueeze(-1)
        # result1 = torch.matmul(v1, E_inv)
        # result2 = torch.matmul(result1, v2).squeeze(-1).squeeze(-1)
        
        exponent = torch.log(alpha) - 0.5 * float(self.output_dim) * torch.log(pi_term) \
            - float(self.output_dim) * torch.log(det) \
            - mahalanobis_sq / 2
        # Replace NaNs with 0
        log_gauss = self.log_sum_exp(exponent, axis=1)
        return -torch.mean(log_gauss) 

    def loss1(self, parameters, y_true):
        # I receive a tensor of shape (batch_size, (output_dim * output_dim + 3 * output_dim + 2) / 2 * num_gaussians)
        # I need to separate the parameters to have:
        # mu: (batch_size, output_dim * num_gaussians)
        # sigma_not_in_diagonal: (batch_size, (output_dim * (output_dim-1)) / 2 * num_gaussians)
        # sigma_diag: (batch_size, output_dim * num_gaussians)
        # alpha: (batch_size, num_gaussians)

        mu = parameters[:, :self.output_dim * self.num_gaussians]
        sigma_not_in_diagonal = parameters[:, self.output_dim * self.num_gaussians:self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2)]
        sigma_diag = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2):self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim]
        alpha = parameters[:, self.output_dim * self.num_gaussians + int(self.num_gaussians * (self.output_dim * (self.output_dim-1)) / 2) + self.num_gaussians * self.output_dim:]

        # Reshape the parameters so I have the correct dimensions
        # Example: mu: (batch_size, output_dim, num_gaussians)
        mu = mu.view(-1,  self.num_gaussians, self.output_dim)
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(-1, self.num_gaussians, int((self.output_dim * (self.output_dim-1)) / 2))
        sigma_diag = sigma_diag.view(-1, self.num_gaussians, self.output_dim)
        alpha = alpha.view(-1, self.num_gaussians)
        
        # # Tensor to store L, filled with zeros
        # lower_triangular = torch.zeros((sigma_diag.shape[0], self.num_gaussians, self.output_dim, self.output_dim), device=parameters.device, dtype=parameters.dtype)

        # # Compute the indices for the lower trinagular portion
        # indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=0, device=parameters.device)
        # # Generate all possible row and column indices
        # rows, cols = torch.meshgrid(torch.arange(self.output_dim), torch.arange(self.output_dim))
        # # Mask the indices to exclude the diagonal and upper triangular portion
        # mask = rows > cols
        # indices = torch.stack((rows[mask], cols[mask]), dim=0)

        # # Reshape sigma_not_in_diagonal to have the correct dimensions
        # sigma_not_in_diagonal = sigma_not_in_diagonal.view(sigma_not_in_diagonal.shape[0], self.num_gaussians, -1)

        # # Fill the lower triangular matrix
        # lower_triangular[:, :, indices[0], indices[1]] = sigma_not_in_diagonal

        # # Add the diagonal to the lower triangular matrix
        # lower_triangular[:, :, torch.arange(self.output_dim), torch.arange(self.output_dim)] = sigma_diag

        # Tensor to store L, filled with zeros
        lower_triangular = torch.zeros((sigma_diag.shape[0], self.num_gaussians, self.output_dim, self.output_dim), device=parameters.device, dtype=parameters.dtype)

        # Compute the indices for the lower triangular portion
        indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=-1, device=parameters.device)

        # Reshape sigma_not_in_diagonal to have the correct dimensions
        sigma_not_in_diagonal = sigma_not_in_diagonal.view(sigma_not_in_diagonal.shape[0], self.num_gaussians, -1)

        # Fill the lower triangular matrix
        lower_triangular[:, :, indices[0], indices[1]] = sigma_not_in_diagonal

        # Add the diagonal to the lower triangular matrix
        lower_triangular[:, :, torch.arange(self.output_dim), torch.arange(self.output_dim)] = nn.ELU()(sigma_diag) + 1

        # Pass sigma_not_in_diagonal and sigma_diag through torch.dist to create a MultivariateNormal distribution
        mvn = dist.MultivariateNormal(mu, scale_tril=lower_triangular)

        # Use alphas to create a mixture of distributions
        mix = dist.Categorical(alpha)
        mixture = dist.MixtureSameFamily(mix, mvn)

        # Calculate the log likelihood
        log_likelihood = mixture.log_prob(y_true)

        # Calculate the mean log likelihood
        return -torch.mean(log_likelihood)



        

    
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
                components = parameters.view(-1, self.output_dim + 2, self.num_gaussians)
                mu_pred = components[:, :self.output_dim, :]
                sigma_pred = components[:, self.output_dim, :]
                alpha_pred = components[:, self.output_dim + 1, :]
                # Sort alphas from highest to lower 
                alpha_pred, indices = torch.sort(alpha_pred, dim=1)
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
    
    def distribution_of_means(self, dataloader, device, num):
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
                components = parameters.view(-1, self.output_dim + 2, self.num_gaussians)
                mu_pred = components[:, :self.output_dim, :]
                sigma_pred = components[:, self.output_dim, :]
                alpha_pred = components[:, self.output_dim+1, :]
                # Sort alphas from highest to lower 
                alpha_pred, indices = torch.sort(alpha_pred, dim=1)
                alpha_pred = alpha_pred.cpu().numpy()
                dim = alpha_pred.shape[1]
                y_pred = np.zeros((len(mu_pred)))  
                y_pred = np.array([mu_pred[i,:,num]  
                    for i in np.arange(len(alpha_pred))])  
                y_pred = torch.tensor(y_pred).to(device)

                # True labels
                y_true_final = torch.cat([y_true_final, labels], dim=0)

                # Predicted labels
                y_pred_final = torch.cat([y_pred_final, y_pred], dim=0)

        return y_true_final, y_pred_final

