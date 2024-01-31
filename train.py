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
from torchsummary import summary

# Important for num_workers > 0
if __name__ == "__main__":
    # Get the arguments from the command line
    args = utils.parser()

    # Get Current Date and Time to name the model
    now = datetime.now()
    current_date = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Dataset
    # Load the data
    full_dataset = StellatorsDataSet(npy_file='data/dataset.npy')

    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    # elif torch.backends.mps.is_available():
    #     device = "mps" # Apple GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")

    # Setup the Hyperparameters
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    MOMENTUM = 0
    NUM_OF_WORKERS = 0

    # Turn datasets into iterable objects (batches)
    # Create DataLoaders with help from data_setup.py
    train_dataloader, val_dataloader, test_dataloader, mean_std = data_setup.create_dataloaders(dataset=full_dataset,
                                                                    train_size=0.5,
                                                                    val_size=0.2,
                                                                    batch_size=BATCH_SIZE,
                                                                    num_workers=NUM_OF_WORKERS
)

    # Create the model from the model_builder.py
    model = model_builder.Model(input_dim=9,
                                output_dim=10
    ).to(device)

    # Set up loss function and optimizer
    loss_fn = getattr(nn, args.loss_function)()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARING_RATE,
                                weight_decay=WEIGHT_DECAY
    )

    # Create the writer for TensorBoard with help from utils.py
    writer = utils.create_writer(experiment_name="MLStellaratorDesign",
                                model_name=model.__class__.__name__,
    )
    # Add Model Architecture to TensorBoard
    writer.add_text("Model Summary", str(model))
    # Add Hyperparameters to TensorBoard
    writer.add_hparams({"batch_size": BATCH_SIZE,
                        "num_epochs": NUM_EPOCHS,
                        "learning_rate": LEARING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "momentum": MOMENTUM,
                        "num_of_workers": NUM_OF_WORKERS,
                        "loss_function": loss_fn.__class__.__name__,
                        "optimizer": optimizer.__class__.__name__,
                        "device": device,
                        "model": model.__class__.__name__,
                        "model_architecture": str(model),
                        "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "model_non_trainable_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
                        "Labels Mean": mean_std["mean_labels"],
                        "Labels Standard Deviation": mean_std["std_labels"],
                        "Features Mean": mean_std["mean"],
                        "Features Standard Deviation": mean_std["std"],

    }, 
    {})


    # Set the seed and start the timer 
    train_time_start_on_gpu = timer()
    print(f"Training on {device}...")

    # Start the training session with engine.py
    engine.train(model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=NUM_EPOCHS,
                device=device,
                classification=False,
                writer=writer)


    # Measure time
    train_time_end_on_gpu = timer()
    total_train_time_model_1 = utils.print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir=f"models/{model.__class__.__name__}",
                    model_name=f"{current_date}.pth")