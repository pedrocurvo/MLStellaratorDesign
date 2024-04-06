import torch 
from qsc import Qsc
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from tabulate import tabulate
sys.path.append('../')

from mdn_torch.MDNFullCovariance import MDNFullCovariance
from mdn_torch.preditcions_utils import sample_output, check_criteria, run_qsc, round_nfp

def predictor(axis_length=None, iota=None, max_elongation=None, min_L_grad_B=None, min_R0=None,
              r_singularity=None, L_grad_grad_B=None, B20_variation=None, beta=None, DMerc_times_r2=None,
              prioritize_DMerc_times_r2_positive=True, iterations=10, device=None):
    """
    Predicts configurations using MDN and Qsc.

    Args:
        axis_length (float, optional): Axis length. Defaults to None.
        iota (float, optional): Iota. Defaults to None.
        max_elongation (float, optional): Maximum elongation. Defaults to None.
        min_L_grad_B (float, optional): Minimum L gradient of magnetic field. Defaults to None.
        min_R0 (float, optional): Minimum major radius. Defaults to None.
        r_singularity (float, optional): Radius of singularity. Defaults to None.
        L_grad_grad_B (float, optional): Gradient of gradient of magnetic field. Defaults to None.
        B20_variation (float, optional): B20 variation. Defaults to None.
        beta (float, optional): Beta. Defaults to None.
        DMerc_times_r2 (float, optional): DMerc times r2. Defaults to None.
        prioritize_DMerc_times_r2_positive (bool, optional): Whether to prioritize positive DMerc times r2. Defaults to True.
        iterations (int, optional): Number of iterations. Defaults to 10.
        device (str, optional): Device to use (cpu or cuda). Defaults to None.

    Returns:
        Qsc: Predicted stellarator configuration.
    """
    # Validate inputs
    if not all(param is None or isinstance(param, (float, int)) for param in [axis_length, iota, max_elongation, min_L_grad_B, min_R0, 
                                                                               r_singularity, L_grad_grad_B, B20_variation, beta, DMerc_times_r2]):
        raise ValueError("All input parameters should be floats or None.")

    # Set device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model_params = {
        'input_dim': 10,
        'output_dim': 10,
        'num_gaussians': 62
    }
    model = MDNFullCovariance(**model_params).to(device)

    # Load models and mean_stds
    models = ["../mdn_torch/models/MDNFullCovariance/2024_03_28_11_53_42.pth",
              "../mdn_torch/models/MDNFullCovariance/2024_03_30_02_40_44.pth",
              "../mdn_torch/models/MDNFullCovariance/2024_04_02_15_47_52.pth",
              "../mdn_torch/models/MDNFullCovariance/2024_04_03_10_03_21.pth",
              "../mdn_torch/models/MDNFullCovariance/2024_04_04_01_37_38.pth",
              "../mdn_torch/models/MDNFullCovariance/2024_04_04_13_41_19.pth",
              "../mdn_torch/models/MDNFullCovariance/2024_04_05_01_01_18.pth"]
    mean_stds = ["../mdn_torch/models/mean_std.pth",
                 "../mdn_torch/models/mean_std_2.pth",
                 "../mdn_torch/models/mean_std_3.pth",
                 "../mdn_torch/models/mean_std_4.pth",
                 "../mdn_torch/models/mean_std_5.pth",
                 "../mdn_torch/models/mean_std_combined.pth",
                 "../mdn_torch/models/mean_std_5.pth"]

    # Initialize lists to store configurations and values for Qsc
    total_configurations = []
    total_values_for_qsc = []
    total_errors = []

    # Iterate over models
    for model_path, mean_std_path in tqdm(zip(models, mean_stds), desc="Models", leave=False):
        # Load model
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        # Load mean_std
        mean_std = torch.load(mean_std_path, map_location=torch.device(device))

        # Sample input
        sample = np.array([axis_length, iota, max_elongation, min_L_grad_B, min_R0, r_singularity, 
                           L_grad_grad_B, B20_variation, beta, DMerc_times_r2], dtype=np.float32)
        sample = np.where(sample == None, mean_std["mean_labels"].cpu(), sample)

        configurations = []
        values_for_qsc = []

        # Run iterations
        for _ in tqdm(range(iterations), leave=False, desc="Iterations"):
            with torch.no_grad():
                # Normalize input
                values = (torch.tensor(sample).float().to(device) - mean_std["mean"].to(device)) / mean_std["std"].to(device)

                # Predict using model
                values = model.getMixturesSample(values.unsqueeze(0), device)

                # Denormalize output
                values = values * mean_std["std_labels"].to(device) + mean_std["mean_labels"].to(device)

                # Run Qsc
                values = values.cpu().numpy()
                values[0] = round_nfp(values[0])
                try:
                    qsc_values = run_qsc(values[0])
                    if prioritize_DMerc_times_r2_positive:
                        if qsc_values[-1] > 0:
                            configurations.append(qsc_values)
                            values_for_qsc.append(values[0])
                    else:
                        continue
                except:
                    continue
        values = torch.tensor(sample).float().to(device)
        errors = []
        for configuration in configurations:
            config_norm = (torch.tensor(configuration).to(device) - mean_std["mean_labels"].to(device)) / mean_std["std_labels"].to(device)
            values_norm = (values - mean_std["mean_labels"].to(device)) / mean_std["std_labels"].to(device)
            errors.append(torch.nn.functional.mse_loss(config_norm, values_norm).item())
        total_errors.extend(errors)
        total_configurations.extend(configurations)
        total_values_for_qsc.extend(values_for_qsc)

    # Select best configuration
    argmin = np.argmin(total_errors)
    best_configuration = total_configurations[argmin]

    # Print results
    print_results(axis_length, iota, max_elongation, min_L_grad_B, min_R0, r_singularity, L_grad_grad_B, 
                  B20_variation, beta, DMerc_times_r2, best_configuration, total_values_for_qsc[argmin])

    # Create and return Qsc object
    return create_qsc_object(total_values_for_qsc[argmin])


def print_results(axis_length, iota, max_elongation, min_L_grad_B, min_R0, r_singularity, L_grad_grad_B, 
                  B20_variation, beta, DMerc_times_r2, program_output, qsc_inputs):
    # Define data
    data = [
        ['axis_length', axis_length, program_output[0], 'rc1', qsc_inputs[0]],
        ['iota', iota, program_output[1], 'rc2', qsc_inputs[1]],
        ['max_elongation', max_elongation, program_output[2], 'rc3', qsc_inputs[2]],
        ['min_L_grad_B', min_L_grad_B, program_output[3] , 'zs1', qsc_inputs[3]],
        ['min_R0', min_R0, program_output[4], 'zs2', qsc_inputs[4]],
        ['r_singularity', r_singularity, program_output[5], 'zs3', qsc_inputs[5]],
        ['L_grad_grad_B', L_grad_grad_B, program_output[6], 'nfp', qsc_inputs[6]],
        ['B20_variation', B20_variation, program_output[7], 'etabar', qsc_inputs[7]],
        ['beta', beta, program_output[8], 'B2c', qsc_inputs[8]],
        ['DMerc_times_r2', DMerc_times_r2, program_output[9], 'p2', qsc_inputs[9]],
    ]

    # Print table
    print(tabulate(data, headers=['Ouput', 'What you asked for', 'Program Output', 'Inputs for QSC', 'Inputs for QSC'], tablefmt='grid'))


def create_qsc_object(program_output):
    # Create Qsc object
    return Qsc(rc=[1., program_output[0], program_output[1], program_output[2]],
               zs=[0., program_output[3], program_output[4], program_output[5]],
               nfp=int(program_output[6]),
               etabar=program_output[7],
               B2c=program_output[8],
               p2=program_output[9],
               order='r2')
