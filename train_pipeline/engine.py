"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter

def train_step(epoch: int,
               model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               classification: bool = True,
               disable_progress_bar: bool = False) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    epoch: An integer indicating the current epoch.
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    classification: A boolean indicating whether the model is a classification
    or regression model.
    disable_progress_bar: A boolean indicating whether to disable the progress bar.

    Returns:
    If classification is True, a tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)

    If classification is False, a single testing loss metric.
    In the form (test_loss). For example:

    (0.0223)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    progress_bar = tqdm(
        enumerate(dataloader), 
        desc=f"Training Epoch {epoch + 1}", 
        total=len(dataloader),
        leave=False,
        disable=disable_progress_bar,
        colour="green"
    )

    # Loop through data loader data batches
    for batch, (X, y) in progress_bar:
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        if classification:
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        # Update progress bar
        progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
            }
        )
        progress_bar.update()

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    if classification:
        return train_loss, train_acc
    else:
        return train_loss

def test_step(epoch: int,
              model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              classification: bool = False,
              disable_progress_bar: bool = False) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    epoch: An integer indicating the current epoch.
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    classification: A boolean indicating whether the model is a classification
    or regression model.
    disable_progress_bar: A boolean indicating whether to disable the progress bar.

    Returns:
    If classification is True, a tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)

    If classification is False, a single testing loss metric.
    In the form (test_loss). For example:

    (0.0223)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Loop through data loader data batches
    progress_bar = tqdm(
        enumerate(dataloader), 
        desc=f"Testing Epoch {epoch + 1}", 
        total=len(dataloader),
        disable=disable_progress_bar,
        leave=False,
        colour="red"
    )

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in progress_bar:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            if classification:
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
            # Update progress bar
            progress_bar.set_postfix(
                {
                  "test_loss": test_loss / (batch + 1),
                }
            )
            progress_bar.update()

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    if classification:
        return test_loss, test_acc
    else:
        return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter = None,
          classification: bool = True, 
          disable_progress_bar: bool = False) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    writer: A SummaryWriter instance to write loss and accuracy metrics to TensorBoard.
    classification: A boolean indicating whether the model is a classification
    or regression model.
    disable_progress_bar: A boolean indicating whether to disable the progress bar.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    progress_bar = tqdm(
        range(epochs),
        desc="Epochs",
        total=epochs,
        disable=disable_progress_bar,
        leave=False,
        colour="blue"
    )

    # Loop through training and testing steps for a number of epochs
    for epoch in progress_bar:
        progress_bar.set_description(f"Epoch {epoch+1}")
        try:
            train_loss, train_acc = 0, 0
            test_loss, test_acc = 0, 0
            if classification:
                train_metrics = train_loss, train_acc
                test_metrics = test_loss, test_acc
            else:
                train_metrics = train_loss 
            train_metrics = train_step(epoch=epoch,
                                            model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device,
                                            classification=classification,
                                            disable_progress_bar=False)
            test_metrics = test_step(epoch=epoch,
                                            model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device,
                                            classification=classification,
                                            disable_progress_bar=False)

            # Print depending on classification or regression
            if classification:
                train_loss, train_acc = train_metrics
                test_loss, test_acc = test_metrics
                # Print out what's happening
                print(
                f"\nEpoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
                )

                # Update results dictionary
                results["train_loss"].append(train_loss)
                results["train_acc"].append(train_acc)
                results["test_loss"].append(test_loss)
                results["test_acc"].append(test_acc)
            
            else:
                train_loss = train_metrics
                test_loss = test_metrics
                # Print out what's happening
                print(
                f"\ntrain_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f}"
                )

                # Update results dictionary
                results["train_loss"].append(train_loss)
                results["test_loss"].append(test_loss)

            if writer:
                # Add loss results to SummaryWriter
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Test", test_loss, epoch)
                if classification:
                    # Add accuracy results to SummaryWriter
                    writer.add_scalar("Accuracy/Train", train_acc, epoch)
                    writer.add_scalar("Accuracy/Test", test_acc, epoch)

                # Track the PyTorch model architecture
                writer.add_graph(model=model, 
                            # Pass in an example input
                            input_to_model=torch.randn(32, 9).to(device))

                # Close the writer
                writer.close()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected.")
            print("Stopping training...")
            break

    # Return the filled results at the end of the epochs
    return results