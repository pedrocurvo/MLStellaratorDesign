from torch.utils.data import DataLoader, random_split
import torch
from torchvision import transforms
import os
from StellaratorDataSet import StellaratorDataSetInverse
# Measure time
from timeit import default_timer as timer
from datetime import datetime
import torch.nn as nn
from train_pipeline import engine, utils, data_setup, predictions
from train_pipeline.MBuilder import MixtureDensityNetwork
from torchsummary import summary
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Important for num_workers > 0
if __name__ == "__main__":
    # Get the arguments from the command line
    args = utils.parser()

    # Get Current Date and Time to name the model
    now = datetime.now()
    current_date = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Dataset
    # Load the data
    full_dataset = StellaratorDataSetInverse(npy_file='data/dataset.npy')

    # full_dataset = full_dataset.calculate_data_counts(
    #                         rc1=0,
    #                         rc2=-10,
    #                         rc3=-10,
    #                         zs1=-10,
    #                         zs2=-10,
    #                         zs3=-10,
    #                           AXIS_LENGTH=0,
    #                           IOTA_MIN = 0.2,
    #                           MAX_ELONGATION = 100,
    #                           MIN_MIN_L_GRAD_B = 0.01,
    #                           MIN_MIN_R0 = 0.3,
    #                           MIN_R_SINGULARITY = 0,
    #                           MIN_L_GRAD_GRAD_B = 0,
    #                           MAX_B20_VARIATION = np.finfo(np.float32).max,
    #                           MIN_BETA = 0,
    #                           MIN_DMERC_TIMES_R2 = -np.finfo(np.float32).max,
    #                           return_object=True)

    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    # elif torch.backends.mps.is_available():
    #     device = "mps" # Apple GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")

    # Setup a Seed for Reproducibility
    torch.manual_seed(0)

    # Setup the Hyperparameters
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
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

    # Create model
    model = MixtureDensityNetwork(input_dim=10,
                                output_dim=10,
                                num_gaussians=5
    ).to(device)

    # Set up loss function and optimizer
    loss_fn = model.mean_log_Laplace_like
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY
    )
    # optimizer=torch.optim.RMSprop(model.parameters(),
    #                               lr=LEARNING_RATE,
    #                               alpha=0.9, eps=1e-07,
    #                               weight_decay=WEIGHT_DECAY,
    #                               momentum=MOMENTUM, centered=False)

    # Create the writer for TensorBoard with help from utils.py
    writer = utils.create_writer(experiment_name=f"{full_dataset.__class__.__name__}",
                                model_name=model.__class__.__name__,
                                timestamp=current_date
    )
    # # Add Model Architecture to TensorBoard
    # writer.add_text("Model Summary", str(model))
    # Add Hyperparameters to TensorBoard
    writer.add_hparams({"batch_size": BATCH_SIZE,
                        "num_epochs": NUM_EPOCHS,
                        "learning_rate": LEARNING_RATE,
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
                    target_dir=f"models/{full_dataset.__class__.__name__}/{model.__class__.__name__}",
                    model_name=f"{current_date}.pth")
    
    # -----------------------------------------------------------------------------
    # Test the model using the test dataset with help from predictions.py

    # True vs Predicted
    y_true, y_pred = model.predict(test_dataloader,
                                    device
    )

    # Loss
    total_test_loss = nn.HuberLossLoss()(y_pred, y_true).item()

    # Add Loss to TensorBoard
    writer.add_scalar("Test Loss", total_test_loss, global_step="HuberLossLoss")

    # Confusion Matrix
    confuse = predictions.nfp_confusion_matrix(
                            y_true=y_true,
                            y_pred=y_pred,
                            mean_labels=mean_std["mean_labels"],
                            std_labels=mean_std["std_labels"]
    )

    # Add Confusion Matrix to TensorBoard
    writer.add_figure("Confusion Matrix", confuse)

    # Add Distributions of Means to TensorBoard
    for i in range(10):
        distribution_hist = predictions.distribution_hist(y_true,
                                                          y_pred,
                                                          full_dataset.labels_names[i],
                                                          i,
                                                          mean_std["mean_labels"][i],
                                                          mean_std["std_labels"][i]
        )
        writer.add_figure(f"Distribution of {full_dataset.labels_names[i]} ", distribution_hist)