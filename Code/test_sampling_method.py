import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli
from scipy.stats.qmc import Sobol as QMC_Sobol, LatinHypercube
import os

def ortho_score(samples):
    R = np.corrcoef(samples, rowvar=False)
    R_off_diag = R - np.diag(np.diag(R))
    ortho_score = np.linalg.norm(R_off_diag, ord='fro')
    return ortho_score

# Setting the output directory
output_file_path = "outputs/test"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# Define the problem: 2 parameters in [0,1] for easy visualization
num_params = 2
num_samples = 300  # Keep it the same for all methods for comparison

# Sobol Sampling (Variance-Based)
problem = {
    'num_vars': num_params,
    'names': [f'x{i+1}' for i in range(num_params)],
    'bounds': [[0, 1]] * num_params
}
sobol_samples = saltelli.sample(problem, num_samples, calc_second_order=False)

# QMC Sobol Sampling (Quasi-Random)
qmc_sobol = QMC_Sobol(d=num_params, scramble=True)
qmc_sobol_samples = qmc_sobol.random(len(sobol_samples))

# Latin Hypercube Sampling
lhs = LatinHypercube(d=num_params)
lhs_samples = lhs.random(len(sobol_samples))

print(f'Size of Sobel: {len(sobol_samples)}')
print(f'Size of QMC: {len(qmc_sobol_samples)}')
print(f'Size LHC: {len(lhs_samples)}')

print('############ \t Orthogonality Scores \t ############')
print(f'Sobol: {ortho_score(sobol_samples)}')
print(f'QMC: {ortho_score(qmc_sobol_samples)}')
print(f'LHC: {ortho_score(lhs_samples)}')

# Function to plot samples
def plot_samples(samples, title, ax):
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10, edgecolor='k')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.5)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_samples(sobol_samples, "Sobol Sampling", axes[0])
plot_samples(qmc_sobol_samples, "QMC Sobol Sampling", axes[1])
plot_samples(lhs_samples, "Latin Hypercube Sampling", axes[2])

plt.tight_layout()
plt.savefig(output_file_path + "/scatter.png")
plt.clf()

# Pairplot comparison
df_sobol = {f'x{i+1}': sobol_samples[:, i] for i in range(num_params)}
df_qmc_sobol = {f'x{i+1}': qmc_sobol_samples[:, i] for i in range(num_params)}
df_lhs = {f'x{i+1}': lhs_samples[:, i] for i in range(num_params)}

# Convert to DataFrames for Seaborn
import pandas as pd
df_sobol = pd.DataFrame(df_sobol)
df_qmc_sobol = pd.DataFrame(df_qmc_sobol)
df_lhs = pd.DataFrame(df_lhs)

# Pairwise comparison plots
sns.pairplot(df_sobol).fig.suptitle("Sobol Sampling - Pairwise Distribution", y=1.02)
sns.pairplot(df_qmc_sobol).fig.suptitle("QMC Sobol Sampling - Pairwise Distribution", y=1.02)
sns.pairplot(df_lhs).fig.suptitle("Latin Hypercube Sampling - Pairwise Distribution", y=1.02)
plt.savefig(output_file_path + "/pair_plot.png")
plt.clf()
