from torch.utils.data import DataLoader, random_split
import torch
import os
from StellatorsDataSet import StellatorsDataSet
# Measure time
from timeit import default_timer as timer
import torch.nn as nn
from train_pipeline import engine, model_builder, utils

if __name__ == "__main__":
    # Dataset
    # Load the data
    full_dataset = StellatorsDataSet(npy_file='data/dataset.npy')

    full_dataset = full_dataset.calculate_data_counts(IOTA_MIN = 0.2,
                                MAX_ELONGATION = 10,
                                MIN_MIN_L_GRAD_B = 0.1,
                                MIN_MIN_R0 = 0.3,
                                MIN_R_SINGULARITY = 0.05,
                                MIN_L_GRAD_GRAD_B = 0.01,
                                MAX_B20_VARIATION = 5,
                                MIN_BETA = 1e-4,
                                MIN_DMERC_TIMES_R2 = 0,
                                return_object=True)

    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    # elif torch.backends.mps.is_available():
    #     device = "mps" # Apple GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")

    print(f"Device: {device}")

    # Setup the Hyperparameters
    BATCH_SIZE = len(full_dataset)
    NUM_EPOCHS = 200000
    LEARING_RATE = 0.001

    # Define sizes for train, validation, and test sets
    train_size = int(0.5 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])



    # Turn datasets into iterable objects (batches)
    train_loader = DataLoader(dataset=train_dataset, # dataset to turn into iterable batches
                            batch_size=BATCH_SIZE, # samples per batch
                            shuffle=True, # shuffle data (for training set only)
                            num_workers=4, # subprocesses to use for data loading
                            pin_memory=True # CUDA only
    ) 

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False
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


    # Start the training session with engine.py
    engine.train(model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
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
                    model_name=f"{model.__class__.__name__}.pth")