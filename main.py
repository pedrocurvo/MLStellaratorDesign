from mdn_torch import qsc_predictor
import pandas as pd 
from qsc import Qsc
# Read CSV 
df = pd.read_csv('./data/XGStels/XGStels.csv')

# Sample a random row
sample = df.sample()

# Run Qsc
stel_original = Qsc(rc=[1., sample['rc1'].values[0], sample['rc2'].values[0], sample['rc3'].values[0]],
                    zs=[0, sample['zs1'].values[0], sample['zs2'].values[0], sample['zs3'].values[0]],
                    nfp=sample['nfp'].values[0],
                    etabar=sample['etabar'].values[0],
                    B2c=sample['B2c'].values[0],
                    p2=sample['p2'].values[0]
)
stel_original.plot_boundary(r=0.05)



if __name__ == "__main__":
    stel = qsc_predictor(axis_length =  sample['axis_length'].values[0],
                        iota =  sample['iota'].values[0],
                        max_elongation =  sample['max_elongation'].values[0],
                        min_L_grad_B =  sample['min_L_grad_B'].values[0],
                        min_R0 =  sample['min_R0'].values[0],
                        r_singularity =  sample['r_singularity'].values[0],
                        L_grad_grad_B =  sample['L_grad_grad_B'].values[0],
                        beta =  sample['beta'].values[0],
                        B20_variation =  sample['B20_variation'].values[0],
                        DMerc_times_r2 =  sample['DMerc_times_r2'].values[0],
                       iterations=10,
                       device=None, 
                       prioritize_DMerc_times_r2_positive=False,
                       print_values=True)
    stel.plot_boundary(r=0.05)

    

