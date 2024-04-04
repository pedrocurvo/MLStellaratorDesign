import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

#fname_0 = '../data/first_dataset.csv'
fname_0 = './data_good/first_good_stels.csv'
print('Reading:', fname_0)
df_0 = pd.read_csv(fname_0)

#fname_1 = '../data/second_dataset.csv'
fname_1 = './data_good/second_good_stels.csv'
print('Reading:', fname_1)
df_1 = pd.read_csv(fname_1)

#fname_2 = '../data/third_dataset.csv'
fname_2 = './data_good/third_good_stels.csv'
print('Reading:', fname_2)
df_2 = pd.read_csv(fname_2)

#fname_3 = '../data/fourth_dataset.csv'
fname_3 = './data_good/fourth_good_stels.csv'
print('Reading:', fname_3)
df_3 = pd.read_csv(fname_3)

#fname_4 = '../data/fifth_dataset.csv'
fname_4 = './data_good/fifth_good_stels.csv'
print('Reading:', fname_4)
df_4 = pd.read_csv(fname_4)

# -----------------------------------------------------------------------------

for col in df_0.columns:
    values_0 = df_0[col].values
    if col == 'axis_lenght':
        values_1 = df_1['axis_length'].values
        values_2 = df_2['axis_length'].values
        values_3 = df_3['axis_length'].values
        values_4 = df_4['axis_length'].values
    else:
        values_1 = df_1[col].values
        values_2 = df_2[col].values
        values_3 = df_3[col].values
        values_4 = df_4[col].values

    if col in ['nfp']:
        vmin = np.min(values_3) - 0.5
        vmax = np.max(values_3) + 0.5
        num = int(round(vmax - vmin + 1))

    # elif col in ['iota', 'r_singularity', 'min_L_grad_B']:
    #     p = np.percentile(np.fabs(values_3), 80)
    #     vmin = max(-p, np.min(values_3))
    #     vmax = min(+p, np.max(values_3))
    #     num = 801

    # elif col in ['max_elongation', 'L_grad_grad_B']:
    #     p = np.percentile(np.fabs(values_3), 50)
    #     vmin = max(-p, np.min(values_3))
    #     vmax = min(+p, np.max(values_3))
    #     num = 501

    # elif col in ['B20_variation', 'iota', 'DMerc_times_r2']:
    #     p = np.percentile(np.fabs(values_3), 10)
    #     vmin = max(-p, np.min(values_3))
    #     vmax = min(+p, np.max(values_3))
    #     num = 101

    else:
        vmin = np.min(values_3)
        vmax = np.max(values_3)
        num = 1001

    bins = np.linspace(vmin, vmax, num=num)

    plt.hist(values_0, bins, density=True, alpha=0.5, label='dataset', log=True)
    plt.hist(values_1, bins, density=True, alpha=0.5, label='pred_1', log=True)
    plt.hist(values_2, bins, density=True, alpha=0.5, label='pred_2', log=True)
    plt.hist(values_3, bins, density=True, alpha=0.5, label='pred_3', log=True)
    plt.hist(values_4, bins, density=True, alpha=0.5, label='pred_4', log=True)

    plt.title(col)
    plt.legend()
    plt.savefig(f'./histograms_good/{col}.png')
    plt.close()
    #plt.show()

# -----------------------------------------------------------------------------
# Plot a correlation matrix
# -----------------------------------------------------------------------------
import seaborn as sns

# Compute the correlation matrix
correlation_matrix = df_3.corr(method='spearman')

# Plot the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix, Dataset3, Good Stels, Method: Spearman')
plt.savefig('./correlations_matrices_good/full_correlation_matrix_good.png')
plt.close()

# -----------------------------------------------------------------------------
# Now plot the correlation matrix for different nfps
# -----------------------------------------------------------------------------
for nfp in range(2, 10):
    df_3_nfp = df_3[df_3['nfp'] == nfp]
    correlation_matrix = df_3_nfp.corr(method='spearman')

    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Correlation Matrix, Dataset3, Good Stels, NFP={nfp}, Method: Spearman')
    plt.savefig(f'./correlations_matrices_good/correlation_matrix_good_nfp_{nfp}.png')
    plt.close()