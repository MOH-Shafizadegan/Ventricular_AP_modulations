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
output_file_path = "outputs/test_sampling"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# Define parameters
num_params = 2
num_samples = 300

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

plot_samples(sobol_samples, f"Sobol - orthogonality = {ortho_score(sobol_samples):.5f}", axes[0])
plot_samples(qmc_sobol_samples, f"QMC Sobol - orthogonality = {ortho_score(qmc_sobol_samples):.5f}", axes[1])
plot_samples(lhs_samples, f"Latin Hypercube - orthogonality = {ortho_score(lhs_samples):.5f}", axes[2])

plt.tight_layout()
plt.savefig(output_file_path + "/scatter.png")
plt.clf()
