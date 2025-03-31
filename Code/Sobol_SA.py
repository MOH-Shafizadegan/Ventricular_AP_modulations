import numpy as np
import os
from Functions.OpenCor_Py.opencor_helper import SimulationHelper
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import qmc
from SALib.analyze import sobol
import SALib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
import seaborn as sns
from SALib.sample import saltelli

from Functions.AP_features import calculate_apd

# Defining functions
def run_and_get_results(param_vals):
    sim_object.set_param_vals(param_names, param_vals)
    sim_object.reset_states()
    sim_object.run()
    
    y = sim_object.get_results(output_names)
    t = sim_object.tSim - pre_time
    return y, t    


working_dir = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(working_dir, "Models/ToRORd_dynCl_endo.cellml")

# Setting the output directory
output_type = 'ADP50'

output_file_path = "outputs/Sobol_SA/" + output_type + "/"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# sim specs
pre_time = 0
sim_time = 1000
dt = 1
sample_type = 'qmc'

# Create simulator object
sim_object = SimulationHelper(model_path, dt, sim_time, solver_info={'MaximumStep':0.001, 'MaximumNumberOfSteps':500000}, pre_time=pre_time)

# Define parametrs
param_names = ['extracellular/ko', 'IKr/GKr_b', 'IK1/GK1_b', 'ICaL/PCa_b']      # This list can be appended
param_vals_mins = [3, 0.002, 0.3, 1.3757e-05]
param_vals_maxs = [5, 0.0121, 0.6992, 8.3757e-05]

# Latin hypercube sampling
problem = {
    'num_vars': len(param_names),
    'names': param_names,
    'bounds': list(zip(param_vals_mins, param_vals_maxs))
}

N = 2000

if sample_type == "sobol":
    samples = saltelli.sample(problem, N, calc_second_order=True)  # Enable second-order interactions

elif sample_type == "qmc":
    sampler = qmc.Sobol(d=problem['num_vars'], scramble=True)  # Scrambled Sobol' for better uniformity
    qmc_samples = sampler.random(N)
    samples = qmc.scale(qmc_samples, [b[0] for b in problem['bounds']], [b[1] for b in problem['bounds']])

print(samples)

output_names = ['membrane/v']


APD90s = []
for i in range(len(samples)):

    current_param_val = samples[i, :]

    # Run the simulation
    y, t = run_and_get_results(current_param_val)
    repolarization_level = int(re.search(r'\d+', output_type).group())
    APD90s.append(calculate_apd(t, np.squeeze(y), repolarization_level, depolarization_threshold=-20))

    print(f"Iteration {i+1}/{len(samples)}.")

Si = sobol.analyze(problem, np.array(APD90s))

# Barplot for first order indices
S1 = Si['S1']
ST = Si['ST']
x = np.arange(len(param_names))
plt.figure(figsize=(8, 5))
plt.bar(x - 0.2, S1, width=0.4, label='First-order', color='blue', alpha=0.7)
plt.bar(x + 0.2, ST, width=0.4, label='Total-order', color='red', alpha=0.7)

plt.xticks(x, param_names)
plt.ylabel('Sensitivity Index')
plt.title(f'Sobol Sensitivity Analysis - output: {output_type}')
plt.legend()
plt.savefig(output_file_path + "/" + sample_type + "_n" + str(N) + "_First_order_idx.png")
plt.clf()

# second order indices
interaction_matrix = np.array(Si['S2'])

plt.figure(figsize=(6, 5))
sns.heatmap(interaction_matrix, annot=True, xticklabels=param_names, yticklabels=param_names, cmap="coolwarm")
plt.title(f"Second-order Sobol Indices (Interactions) - output: {output_type}")
plt.savefig(output_file_path + "/" + sample_type + "_n" + str(N) + "_second_order_idx.png")
plt.clf()