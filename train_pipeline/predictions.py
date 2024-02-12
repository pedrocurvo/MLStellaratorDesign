"""
Utility functions to make predictions and evaluate the model.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchinfo import summary
from tqdm import tqdm
import io

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


def nfp_confusion_matrix(y_true: torch.tensor,
                         y_pred: torch.tensor,
                         nfp_index: int = 6,
                         mean_labels: torch.tensor = 0,
                         std_labels: torch.tensor = 1,
                         normalize: int = 0):
    """Returns a confusion matrix for the nfp predictions.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for the test set.
        model (torch.nn.Module): The model to make predictions with.
        device (torch.device): The device to run the model on.
        mean (torch.tensor): The mean of the training dataset. Defaults to 0.
        std (torch.tensor): The standard deviation of the training dataset. Defaults to 1.
        mean_labels (torch.tensor): The mean of the training dataset labels. Defaults to 0.
        std_labels (torch.tensor): The standard deviation of the training dataset labels. Defaults to 1.
        normalize (int, optional): The normalization dimension. Defaults to 0 (col normalization), 1 for row normalization.

    Returns:
        torch.tensor: The confusion matrix.
    """

    # Create an empty confusion matrix
    matrix = torch.zeros(10, 10, dtype=torch.int32)

    # Renormalize the predictions
    y_pred = y_pred[:, nfp_index] * std_labels[nfp_index] + mean_labels[nfp_index]
    y_pred = torch.round(y_pred).detach().numpy().astype(int)

    y_true = y_true[:, nfp_index] * std_labels[nfp_index] + mean_labels[nfp_index]
    y_true = torch.round(y_true).detach().numpy().astype(int)

    # Create a progress bar
    progress_bar = tqdm(
        enumerate(y_pred),
        total=len(y_pred),
        desc="Creating Confusion Matrix",
        leave=False,
        colour="green"
    )
    # Accumulate the confusion matrix
    for i, _ in progress_bar:
        if 0 <= y_true[i] - 1 < 10 and 0 <= y_pred[i] - 1 < 10:
            matrix[y_true[i] - 1, y_pred[i] - 1] += 1
        progress_bar.update()
        
    # Metrics for Matrix
    total_samples = torch.sum(matrix)
    percentage_of_acceptance = total_samples / y_pred.shape[0]
    accuracy = torch.trace(matrix) / torch.sum(matrix)
    precision = torch.diag(matrix) / torch.sum(matrix, dim=0)
    recall = torch.diag(matrix) / torch.sum(matrix, dim=1)
    f1 = 2 * precision * recall / (precision + recall)
    metrics = {"accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1}
        
    print(f"Total Samples: {total_samples}")
    print(f"Percentage of Acceptance: {percentage_of_acceptance}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    
    # Normalize the confusion matrix
    matrix = matrix / matrix.sum(axis=normalize, keepdims=True)
    # Only keep 2 decimals
    matrix = matrix.detach().numpy()
    matrix = np.around(matrix, decimals=2)
    
    # Create an Image of the Confusion Matrix
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(matrix, annot=True, cmap="Blues")
    plt.title("NFP Confusion Matrix")
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    tick_marks = np.arange(10) + 0.5
    # NFP go from 1 to 10, so we need to shift the ticks
    plt.xticks(tick_marks, np.arange(1, 11))
    plt.yticks(tick_marks, np.arange(1, 11))

    return figure


def distribution_hist(y_true: torch.tensor,
                      y_pred: torch.tensor,
                      variable_name: str,
                      variable_index: int,
                      mean: torch.tensor = 0,
                      std: torch.tensor = 1):
    """Returns a histogram for the distribution of a variable."""

    # Renormalize the predictions
    y_pred = y_pred[:, variable_index] * std + mean
    if variable_index == 6:
        y_pred = torch.round(y_pred)
    y_pred = y_pred.detach().numpy()

    y_true = y_true[:, variable_index] * std + mean
    y_true = y_true.detach().numpy()


    # Create a figure
    figure = plt.figure(figsize=(8, 8))
    sns.histplot(y_true, color="blue", kde=True, label="True", stat="density")
    sns.histplot(y_pred, color="red", kde=True, label="Predicted", stat="density")
    plt.xlim(np.min(y_true), np.max(y_true))
    plt.legend()
    plt.title(f"Distribution of {variable_name}")

    return figure

