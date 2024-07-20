import torch 
from qsc import Qsc
from tqdm import tqdm
import numpy as np
import os 
from tabulate import tabulate

from .MDNFullCovariance import MDNFullCovariance
from .utils import run_qsc, round_nfp

def qsc_predictor(
    axis_length=None,
    iota=None,
    max_elongation=None,
    min_L_grad_B=None,
    min_R0=None,
    r_singularity=None,
    L_grad_grad_B=None,
    B20_variation=None,
    beta=None,
    DMerc_times_r2=None,
    prioritize_DMerc_times_r2_positive=True,
    iterations=10,
    choosen_model=None,
    loss_function="mse",
    device=None,
    print_values=True,
):

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
    if not all(param is None or isinstance(param, (float, int)) for param in [axis_length,
                                                                              iota,
                                                                              max_elongation,
                                                                              min_L_grad_B,
                                                                              min_R0, 
                                                                              r_singularity,
                                                                              L_grad_grad_B,
                                                                              B20_variation,
                                                                              beta,
                                                                              DMerc_times_r2]):
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
    models = [ #"../mdn_torch/models/MDNFullCovariance/2024_03_28_11_53_42.pth",
    #           "../mdn_torch/models/MDNFullCovariance/2024_03_30_02_40_44.pth",
    #           "../mdn_torch/models/MDNFullCovariance/2024_04_02_15_47_52.pth",
    #           "../mdn_torch/models/MDNFullCovariance/2024_04_03_10_03_21.pth",
    #           "../mdn_torch/models/MDNFullCovariance/2024_04_04_01_37_38.pth",
    #           "../mdn_torch/models/MDNFullCovariance/2024_04_04_13_41_19.pth",
              "./models/MDNFullCovariance/2024_04_05_01_01_18.pth"]
    mean_stds = [   # "../mdn_torch/models/mean_std.pth",
                    #  "../mdn_torch/models/mean_std_2.pth",
                    #  "../mdn_torch/models/mean_std_3.pth",
                    #  "../mdn_torch/models/mean_std_4.pth",
                    #  "../mdn_torch/models/mean_std_5.pth",
                    #  "../mdn_torch/models/mean_std_combined.pth",
                 "./models/mean_std_5.pth"]

    # Initialize lists to store configurations and values for Qsc
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, models[0])
    mean_std_path = os.path.join(current_dir, mean_stds[0])

    # Load model
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # Load mean_std
    mean_std = torch.load(mean_std_path, map_location=torch.device(device))

    # Sample input
    model_input = np.array([axis_length,
                            iota,
                            max_elongation,
                            min_L_grad_B,
                            min_R0,
                            r_singularity, 
                            L_grad_grad_B,
                            B20_variation,
                            beta,
                            DMerc_times_r2],
                            dtype=np.float32)

    qsc_inputs_to_keep = None
    qsc_output_to_keep = None
    loss_error = None

    # Normalize input
    model_inputs_normalized = (torch.tensor(model_input).float().to(device) - mean_std["mean"].to(device)) / mean_std["std"].to(device)
    if iterations == 0:
        with torch.no_grad():
            # Predict using model
            model_outputs = model.getMixturesSample(model_inputs_normalized.unsqueeze(0), device)

            # Denormalize output
            model_outputs = model_outputs * mean_std["std_labels"].to(device) + mean_std["mean_labels"].to(device)

            # Run Qsc
            model_outputs = model_outputs.cpu().numpy()
            model_outputs[0] = round_nfp(model_outputs[0])
            model_outputs = model_outputs[0]
            return create_qsc_object(model_outputs)



    # Run iterations
    for it in tqdm(range(iterations), leave=False, desc="Iterations"):
        with torch.no_grad():
            try: 
                # Predict using model
                model_outputs = model.getMixturesSample(model_inputs_normalized.unsqueeze(0), device)
            except:
                continue

        # Denormalize output
        model_outputs = model_outputs * mean_std["std_labels"].to(device) + mean_std["mean_labels"].to(device)

        # Run Qsc
        model_outputs = model_outputs.cpu().numpy()[0]
        model_outputs = round_nfp(model_outputs)
        try:
            # Run Qsc
            qsc_output = run_qsc(model_outputs)
            # Normalize qsc_values
            qsc_output_normalized = (torch.tensor(qsc_output).to(device) - mean_std["mean"].to(device)) / mean_std["std"].to(device)
            # Calculate error
            error = torch.nn.functional.mse_loss(model_inputs_normalized, qsc_output_normalized).item()

            if it == 0:
                qsc_inputs_to_keep = model_outputs
                qsc_output_to_keep = qsc_output
                loss_error = error
            else:
                if prioritize_DMerc_times_r2_positive:
                    if qsc_inputs_to_keep[-1] < 0:
                        qsc_inputs_to_keep = model_outputs
                        qsc_output_to_keep = qsc_output
                        loss_error = error
                    elif qsc_output[-1] > 0 and error < loss_error:
                        qsc_inputs_to_keep = model_outputs
                        qsc_output_to_keep = qsc_output
                        loss_error = error
                elif error < loss_error:
                    qsc_inputs_to_keep = model_outputs
                    qsc_output_to_keep = qsc_output
                    loss_error = error
        except:
            continue
    # Print results
    if print_values: 
        print_results(axis_length, iota, max_elongation, min_L_grad_B, min_R0, r_singularity, L_grad_grad_B, 
                    B20_variation, beta, DMerc_times_r2, qsc_output_to_keep, qsc_inputs_to_keep)

    # Create and return Qsc object
    return create_qsc_object(qsc_inputs_to_keep)

# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
def create_qsc_object(program_output):
    # Create Qsc object
    return Qsc(rc=[1., program_output[0], program_output[1], program_output[2]],
               zs=[0., program_output[3], program_output[4], program_output[5]],
               nfp=int(program_output[6]),
               etabar=program_output[7],
               B2c=program_output[8],
               p2=program_output[9],
               order='r2')