"""
Utility functions to make predictions and evaluate the model.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_builder import Model
from sklearn.metrics import confusion_matrix
from torchinfo import summary

def predict(model, data, device):
    """
    Make predictions on a batch of data.

    Args:
        model: PyTorch model.
        data: Batch of data to make predictions on.
        device: Device to run the model on.

    Returns:
        predictions: Predictions made by the model.
    """
    # Set the model to evaluation mode
    model.eval()

    # Turn off gradients
    with torch.no_grad():
        # Move the data to the device
        data = data.to(device)

        # Make predictions
        predictions = model(data)

    return predictions

def correlation_coefficient(model, data, device, method="pearson"):
    """
    Calculate the correlation coefficient between the predictions and the targets.

    Args:
        predictions: Predictions made by the model.
        targets: Targets from the dataset.

    Returns:
        correlation_coefficient: Correlation coefficient between the predictions and the targets.
    """
    # Calculate predictions from Model
    predictions = predict(model, data, device)
    # Convert predictions and targets to NumPy arrays
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Show Correlation Matrix
    df = pd.DataFrame(np.c_[predictions, targets], columns=["predictions", "targets"])

    # Compute the correlation matrix for the inputs 
    correlation_matrix = df.corr(method=method)

    # Plot the correlation matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues")
    plt.title("Correlation Matrix")
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.show()
    return correlation_coefficient


# Testar
model = Model(9, 10)
# Load the model
model.load_state_dict(torch.load("../models/Model.pth"))
# Set the model to evaluation mode
model.eval()
# Load the data
data = np.load("../data/dataset.npy")
# Convert data to tensor
data = torch.from_numpy(data).float()
# Separate the data into inputs and targets
inputs = data[:, :9]
print(f'Size of inputs: {inputs.shape}')
targets = data[:, 9:]
print(f'Size of targets: {targets.shape}')
# Make predictions
predictions = predict(model, inputs, "cpu")
print(f'Size of predictions: {predictions.shape}')
# # Convert predictions and targets to NumPy arrays
# predictions = predictions.cpu().numpy()
# targets = targets.cpu().numpy()
# # With predictions and targets, plot the correlation matrix
# df_predictions = pd.DataFrame(predictions)
# df_targets = pd.DataFrame(targets)

# # Convert dataframes to series if they're not already
# predictions = df_predictions.squeeze()
# targets = df_targets.squeeze()

# Check MSE for first column
for i in range(9):
    mse = F.mse_loss(predictions[:,i], targets[:,i])
    print(f'MSE for column {i}: {mse}')
    if i == 6:
        pred = torch.round(predictions[:,i])
        mse = F.mse_loss(pred, targets[:,i])
        print(f'MSE for column {i} rounded: {mse}')


