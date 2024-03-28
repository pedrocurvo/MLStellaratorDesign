"""
Contains various utility functions for PyTorch model training and saving.
"""
from timeit import default_timer as timer 
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

def parser():
    """Parses arguments from the command line.

    Returns:
        argparse.Namespace: Arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Train a model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", "--epochs" ,type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--loss_function",
                        type=str,
                        default="MSELoss",
                        choices=["L1Loss", "MSELoss", "HuberLoss", "SmoothL1Loss"],
                        help="Loss function")
    args = parser.parse_args()
    return args


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

def norm(X, mean, std):
    """Normalizes a tensor X with mean and std.

    Args:
        X (torch.Tensor): A tensor to normalize.
        mean (torch.Tensor): The mean of X.
        std (torch.Tensor): The standard deviation of X.

    Returns:
        torch.Tensor: A normalized tensor X.
    """
    return (X - mean) / std

def create_writer(experiment_name: str, 
                  model_name: str, 
                  timestamp: str,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", experiment_name, model_name, timestamp, extra)
    else:
        log_dir = os.path.join("runs", experiment_name, model_name, timestamp)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def set_dataset_statistics(dataset, mean, mean_labels, std, std_labels, max_val, max_labels, min_val, min_labels):
    dataset.mean = mean
    dataset.mean_labels = mean_labels
    dataset.std = std
    dataset.std_labels = std_labels
    dataset.max = max_val
    dataset.max_labels = max_labels
    dataset.min = min_val
    dataset.min_labels = min_labels