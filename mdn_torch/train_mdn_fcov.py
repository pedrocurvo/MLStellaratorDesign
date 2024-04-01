import torch
import torch_optimizer as optim

# Add Parent Directory to Python Path
# Inside your Python script within the external_package directory
import sys
import os

from StellaratorDataSet import StellaratorDataSetInverse
# Measure time
from timeit import default_timer as timer
from datetime import datetime
from train_pipeline import engine, utils, data_setup
from MDNFullCovariance import MDNFullCovariance

# Important for num_workers > 0
if __name__ == "__main__":
    # Get the arguments from the command line
    args = utils.parser()

    # Get Current Date and Time to name the model
    now = datetime.now()
    current_date = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Download the dataset
    #data_setup.download_data('../data/second_dataset.csv')

    # Dataset
    # Load the data
    full_dataset = StellaratorDataSetInverse(npy_file='../data/second_dataset.npy')

    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")

    # Setup a Seed for Reproducibility
    torch.manual_seed(torch.randint(0, 100000, (1,)).item())

    # Setup the Hyperparameters
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    MOMENTUM = 0
    NUM_OF_WORKERS = os.cpu_count()

    # Turn datasets into iterable objects (batches)
    # Create DataLoaders with help from data_setup.py
    train_dataloader, val_dataloader, test_dataloader, mean_std = data_setup.create_dataloaders(dataset=full_dataset,
                                                                    train_size=0.5,
                                                                    val_size=0.2,
                                                                    batch_size=BATCH_SIZE,
                                                                    num_workers=NUM_OF_WORKERS
)   
    # Save the mean_std with the same name as the model
    torch.save(mean_std, f"models/mean_std_{current_date}.pth")

    # Create model
    model = MDNFullCovariance(input_dim=10,
                            output_dim=10,
                            num_gaussians=62
    ).to(device)

    # Initialize the weights of the model
    for name, param in model.named_parameters():
        if "weight" in name:
            torch.nn.init.xavier_normal_(param)
        if "bias" in name:
            torch.nn.init.zeros_(param)
    
    # Load a previous model (optional: uncomment if you want to load a previous model): transfer learning
    #model.load_state_dict(torch.load("models/MDNFullCovariance/2024_03_28_11_53_42.pth"))

    # Set up loss function and optimizer
    loss_fn = model.log_prob_loss
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE,
                                eps=1e-08,
                                amsgrad=True,
                                weight_decay=0
    )

    # optimizer = optim.Adahessian(model.parameters(),
    #                             lr= LEARNING_RATE,
    #                             betas= (0.9, 0.999),
    #                             eps= 1e-8,
    #                             weight_decay=0.0,
    #                             hessian_power=0.5,
    # )   
    # optimizer=torch.optim.RMSprop(model.parameters(),
    #                               lr=LEARNING_RATE,
    #                               alpha=0.9, eps=1e-07,
    #                               weight_decay=0,
    #                               momentum=MOMENTUM, centered=False)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,100,300], gamma=0.1)


    # Create the writer for TensorBoard with help from utils.py
    writer = utils.create_writer(experiment_name=f"{full_dataset.__class__.__name__}",
                                model_name=model.__class__.__name__,
                                timestamp=current_date
    )

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
                writer=writer,
                learning_rate_scheduler=scheduler)


    # Measure time
    train_time_end_on_gpu = timer()
    total_train_time_model_1 = utils.print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir=f"models/{model.__class__.__name__}",
                    model_name=f"{current_date}.pth")

    # Close the writer
    writer.close()