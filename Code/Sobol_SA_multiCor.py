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
import pandas as pd
from Functions.AP_features import calculate_apd
import json
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm


# ---------- Simulation setup ----------

# Paths
working_dir = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(working_dir, "Models/ToRORd_dynCl_endo.cellml")

# Output config
output_type = 'APD90'
date_time = datetime.now().isoformat()
output_file_path = os.path.join("outputs", "Sobol_SA", output_type, date_time)
os.makedirs(output_file_path, exist_ok=True)

# Simulation specs
pre_time = 0
sim_time = 750
dt = 1
sample_type = 'sobol'

# Parameter space
param_names = ['extracellular/ko', 'IKr/GKr_b', 'ICaL/PCa_b']
param_vals_maxs = [5, 0.0121, 8.3757e-05]
param_vals_mins = [0.5*val for val in param_vals_maxs]

# Problem definition for SALib
problem = {
    'num_vars': len(param_names),
    'names': param_names,
    'bounds': list(zip(param_vals_mins, param_vals_maxs))
}

# Number of samples
N = 1000

# Sampling
if sample_type == "sobol":
    samples = saltelli.sample(problem, N, calc_second_order=True)
elif sample_type == "qmc":
    sampler = qmc.Sobol(d=problem['num_vars'], scramble=True)
    qmc_samples = sampler.random(N)
    samples = qmc.scale(qmc_samples, [b[0] for b in problem['bounds']], [b[1] for b in problem['bounds']])
else:
    raise ValueError(f"Unknown sample type: {sample_type}")

# Outputs to record from simulation
output_names = ['membrane/v']

# Create simulator object globally (to be re-initialized in each process)
def init_simulator():
    global sim_object
    sim_object = SimulationHelper(
        model_path,
        dt,
        sim_time,
        solver_info={'MaximumStep': 0.001, 'MaximumNumberOfSteps': 500000},
        pre_time=pre_time
    )

# Worker function for parallel processing
def compute_apd(i):
    current_param_val = samples[i, :]
    sim_object.set_param_vals(param_names, current_param_val)
    sim_object.reset_states()
    sim_object.run()
    y = sim_object.get_results(output_names)
    t = sim_object.tSim - pre_time
    rep_level = int(re.search(r'\d+', output_type).group())
    apd = calculate_apd(t, np.squeeze(y), rep_level, depolarization_threshold=-20)
    # print(f"Iteration {i + 1}/{len(samples)}.")
    return apd

# ---------- Parallel execution ----------
n_core = 20
if __name__ == "__main__":
    print(f"Running {N} simulations using {n_core} CPU cores...")

    with Pool(processes=n_core, initializer=init_simulator) as pool:
        results = list(tqdm(pool.imap(compute_apd, range(len(samples))), total=len(samples)))

    APD90s = results

    # Save results to CSV
    df = pd.DataFrame(samples, columns=param_names)
    df[output_type] = APD90s

    csv_path = os.path.join(output_file_path, f"{output_type}_results.csv")
    df.to_csv(csv_path, index=False)

    # Save config
    cfg = {
        "output_type": output_type,
        "sample_type": sample_type,
        "num_samples": N,
        "pre_time": pre_time,
        "sim_time": sim_time,
        "dt": dt,
        "param_names": param_names,
        "param_vals_mins": param_vals_mins,
        "param_vals_maxs": param_vals_maxs,
        "solver_info": {
            "MaximumStep": 0.001,
            "MaximumNumberOfSteps": 500000
        },
        "output_names": output_names
    }

    cfg_path = os.path.join(output_file_path, "config.json")
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=4)

    # ---------- Sobol analysis ----------
    Si = sobol.analyze(problem, np.array(APD90s))

    # Bar plot of first and total order indices
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
    plt.savefig(os.path.join(output_file_path, f"{output_type}_{sample_type}_n{N}_First_order_idx.png"))
    plt.clf()

    # Heatmap of second-order indices
    interaction_matrix = np.array(Si['S2'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(interaction_matrix, annot=True, xticklabels=param_names, yticklabels=param_names, cmap="coolwarm")
    plt.title(f"Second-order Sobol Indices (Interactions) - output: {output_type}")
    plt.savefig(os.path.join(output_file_path, f"{output_type}_{sample_type}_n{N}_second_order_idx.png"))
    plt.clf()
