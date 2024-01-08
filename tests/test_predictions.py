from train_pipeline.model_builder import Model
import torch
from qsc import Qsc
from qsc.util import mu0, fourier_minimum

loaded_model = Model(input_dim=9,
                        output_dim=10)
loaded_model.load_state_dict(torch.load('models/05_going_modular_script_mode_tinyvgg_model.pth'))

loaded_model.eval()
with torch.inference_mode():
    X_test = [[0.25, 9, 0.23, 0.4, 0.06, 0.11, 1, 0.0002, 1]]
    X_test = torch.tensor(X_test).float()
    loaded_model_preds = loaded_model(X_test)
    print(loaded_model_preds)

    rc = [1., loaded_model_preds[0][0].item(), loaded_model_preds[0][1].item(), loaded_model_preds[0][2].item()]
    zs = [0., loaded_model_preds[0][3].item(), loaded_model_preds[0][4].item(), loaded_model_preds[0][5].item()]

    nfp = round(loaded_model_preds[0][6].item())

    etabar = loaded_model_preds[0][7].item()

    B2c = loaded_model_preds[0][8].item()

    p2 = loaded_model_preds[0][9].item()

    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar, B2c=B2c, p2=p2, order='r2')
    iota  = stel.iota
    max_elongation = stel.max_elongation
    min_L_grad_B   = stel.min_L_grad_B
    min_R0         = stel.min_R0
    r_singularity  = stel.r_singularity
    L_grad_grad_B  = fourier_minimum(stel.L_grad_grad_B)
    B20_variation  = stel.B20_variation
    beta           = -mu0 * p2 * stel.r_singularity**2 / stel.B0**2
    DMerc_times_r2 = stel.DMerc_times_r2

    print(f'iota: {iota}')
    print(f'max_elongation: {max_elongation}')
    print(f'min_L_grad_B: {min_L_grad_B}')
    print(f'min_R0: {min_R0}')
    print(f'r_singularity: {r_singularity}')
    print(f'L_grad_grad_B: {L_grad_grad_B}')
    print(f'B20_variation: {B20_variation}')
    print(f'beta: {beta}')
    print(f'DMerc_times_r2: {DMerc_times_r2}')

