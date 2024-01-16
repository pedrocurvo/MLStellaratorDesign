from torch.utils.data import DataLoader, random_split
import torch
from torchvision import transforms
import os
from StellatorsDataSet import StellatorsDataSet
# Measure time
from timeit import default_timer as timer
from datetime import datetime
import torch.nn as nn
from train_pipeline import engine, model_builder, utils, data_setup

# Important for num_workers > 0
if __name__ == "__main__":
    # Get Current Date and Time to name the model
    now = datetime.now()
    current_date = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Dataset
    # Load the data
    full_dataset = StellatorsDataSet(npy_file='data/dataset.npy', sample_size=90000)

    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    # elif torch.backends.mps.is_available():
    #     device = "mps" # Apple GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")

    # Setup the Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARING_RATE = 0.1

    # Turn datasets into iterable objects (batches)
    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader = data_setup.create_dataloaders(dataset=full_dataset,
                                                                    train_size=0.7,
                                                                    batch_size=BATCH_SIZE
)

    # Create the model from the model_builder.py
    model = model_builder.Model(input_dim=9,
                                output_dim=10
    ).to(device)

    # Set up loss function and optimizer
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARING_RATE)

    # Set the seed and start the timer 
    train_time_start_on_gpu = timer()
    print(f"Training on {device}...")

    # Start the training session with engine.py
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=NUM_EPOCHS,
                device=device,
                classification=False)


    # Measure time
    train_time_end_on_gpu = timer()
    total_train_time_model_1 = utils.print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name=f"{model.__class__.__name__}_{current_date}.pth")