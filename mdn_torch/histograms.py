import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

fname_0 = '../data/first_dataset.csv'
#fname_0 = './data_good/first_good_stels.csv'
print('Reading:', fname_0)
df_0 = pd.read_csv(fname_0)

fname_1 = '../data/second_dataset.csv'
#fname_1 = './data_good/second_good_stels.csv'
print('Reading:', fname_1)
df_1 = pd.read_csv(fname_1)

fname_2 = '../data/third_dataset.csv'
#fname_2 = './data_good/third_good_stels.csv'
print('Reading:', fname_2)
df_2 = pd.read_csv(fname_2)

# -----------------------------------------------------------------------------

for col in df_0.columns:
    values_0 = df_0[col].values
    if col == 'axis_lenght':
        values_1 = df_1['axis_length'].values
        values_2 = df_2['axis_length'].values
    else:
        values_1 = df_1[col].values
        values_2 = df_2[col].values

    if col in ['nfp']:
        vmin = np.min(values_0) - 0.5
        vmax = np.max(values_0) + 0.5
        num = int(round(vmax - vmin + 1))

    elif col in ['iota', 'r_singularity', 'min_L_grad_B']:
        p = np.percentile(np.fabs(values_0), 80)
        vmin = max(-p, np.min(values_0))
        vmax = min(+p, np.max(values_0))
        num = 801

    elif col in ['max_elongation', 'L_grad_grad_B']:
        p = np.percentile(np.fabs(values_0), 50)
        vmin = max(-p, np.min(values_0))
        vmax = min(+p, np.max(values_0))
        num = 501

    elif col in ['B20_variation', 'beta', 'DMerc_times_r2']:
        p = np.percentile(np.fabs(values_0), 10)
        vmin = max(-p, np.min(values_0))
        vmax = min(+p, np.max(values_0))
        num = 101

    else:
        vmin = np.min(values_0)
        vmax = np.max(values_0)
        num = 1001

    bins = np.linspace(vmin, vmax, num=num)

    plt.hist(values_0, bins, density=True, alpha=0.5, label='dataset')
    plt.hist(values_1, bins, density=True, alpha=0.5, label='pred_1')
    plt.hist(values_2, bins, density=True, alpha=0.5, label='pred_2')

    plt.title(col)
    plt.legend()
    plt.savefig(f'./histograms/{col}.png')
    plt.close()
    #plt.show()