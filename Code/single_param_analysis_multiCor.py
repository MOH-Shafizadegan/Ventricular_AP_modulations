import opencor as oc
import numpy as np
import random
import os
from Functions.OpenCor_Py.opencor_helper import SimulationHelper
import matplotlib
import matplotlib.pyplot as plt
from SALib.sample import latin
from scipy.stats import qmc
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json
from Functions.AP_features import calculate_apd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Configuration
cfg = {
    "output_feature": 'APD90',
    "sample_type": 'qmc',
    "num_samples": 300,
    "pre_time": 0,
    "sim_time": 800,
    "dt": 0.1,
    "cell_model": 'ToRORd_dynCl_endo.cellml',
    "solver_info": {
        "MaximumStep": 0.001,
        "MaximumNumberOfSteps": 500000
    }
}

# Parameter definitions
cfg["param_names"] = ['extracellular/ko', 'IKr/GKr_b', 'ICaL/PCa_b']
cfg["param_vals_maxs"] = [5, 0.0121, 8.3757e-05]
cfg["param_vals_mins"] = [0.5 * val for val in cfg["param_vals_maxs"]]
cfg["output_names"] = ['membrane/v']

# Output setup
output_feature = cfg["output_feature"]
date_time = datetime.now().isoformat()
output_file_path = os.path.join("outputs/Single_param_analysis", output_feature, date_time)
os.makedirs(output_file_path, exist_ok=True)

cfg_path = os.path.join(output_file_path, "config.json")
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=4)

repolarization_level = int(re.search(r'\d+', output_feature).group())

# Model and simulator setup
working_dir = os.path.dirname(__file__)
model_path = os.path.join(working_dir, "Models", cfg["cell_model"])
sim_object = SimulationHelper(model_path, cfg["dt"], cfg["sim_time"], solver_info=cfg["solver_info"], pre_time=cfg["pre_time"])

# Define simulation runner
def run_and_get_results(param_vals):
    sim_object.set_param_vals(cfg["param_names"], param_vals)
    sim_object.reset_states()
    sim_object.run()
    y = sim_object.get_results(cfg["output_names"])
    t = sim_object.tSim - cfg["pre_time"]
    return y, t

# Sampling
problem = {
    'num_vars': len(cfg["param_names"]),
    'names': cfg["param_names"],
    'bounds': list(zip(cfg["param_vals_mins"], cfg["param_vals_maxs"]))
}

if cfg["sample_type"] == "lhs":
    samples = latin.sample(problem, cfg["num_samples"])
elif cfg["sample_type"] == "qmc":
    sampler = qmc.Sobol(d=problem['num_vars'], scramble=True)
    qmc_samples = sampler.random(cfg["num_samples"])
    samples = qmc.scale(qmc_samples, [b[0] for b in problem['bounds']], [b[1] for b in problem['bounds']])

# Original AP
init_param_vals = sim_object.get_init_param_vals(cfg["param_names"])
y, t = run_and_get_results(init_param_vals)
orig_out = np.squeeze(y)
orig_out_feature = calculate_apd(t, orig_out, repolarization_level, depolarization_threshold=-20)

# Parallel execution function
def simulate_single_param_change(args):
    i, sample_row, init_param_vals = args
    current_param_val = init_param_vals.copy()
    current_param_val[i] = sample_row[i]
    y, t = run_and_get_results(current_param_val)
    v = np.squeeze(y)
    feature = calculate_apd(t, v, repolarization_level, depolarization_threshold=-20)
    return v, feature

# Plot settings
is_plot_sample_AP = 1
is_plot_scatter = 1
n_out_plot = 5

# Main simulation loop
for i, param in enumerate(cfg["param_names"]):
    sim_object.set_param_vals(cfg["param_names"], init_param_vals)

    args_list = [(i, samples[s], init_param_vals) for s in range(len(samples))]

    with ProcessPoolExecutor(max_workers=15) as executor:
        results = list(tqdm(executor.map(simulate_single_param_change, args_list),
                            total=len(args_list), desc=f"Processing {param}"))

    outs, out_feature = zip(*results)

    match = re.search(r'(?<=/)(.*)', str(param))
    param_label = match.group(1)

    if is_plot_sample_AP:
        selected_indices = random.sample(range(len(outs)), min(n_out_plot, len(outs)))
        plt.plot(t, orig_out, linewidth=2, color='black', linestyle='--', label=f'Original AP \n {param_label} = {init_param_vals[i]:.7f}')
        for j in selected_indices:
            plt.plot(t, outs[j], label=f'{param_label} = {samples[j,i]:.7f}')

        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.grid(False)
        plt.legend()
        plt.savefig(os.path.join(output_file_path, param_label + "_AP.png"))
        plt.clf()

    if is_plot_scatter:
        x_min, x_max = np.min(samples[:,i]), np.max(samples[:,i])
        x_norm = samples[:,i] / x_max
        y_norm = out_feature

        model = LinearRegression()
        x_reshape = x_norm.reshape(-1, 1)
        model.fit(x_reshape, y_norm)
        slope = model.coef_[0]
        intercept = model.intercept_
        y_pred = model.predict(x_reshape)

        init_x_norm = init_param_vals[i] / x_max
        init_y_norm = orig_out_feature

        plt.figure(figsize=(6, 5))
        plt.scatter(init_x_norm, init_y_norm, c='black', marker='*')
        plt.scatter(x_norm, y_norm, c='r', marker='o')
        plt.plot(x_norm, y_pred, color='blue', label='Fitted line')
        equation_text = f"y = {slope:.5f}x + {intercept:.5f}"
        plt.text(0.95, 0.95, equation_text, transform=plt.gca().transAxes,
                 ha='right', va='top', fontsize=18, color='black')
        plt.xlabel(param, fontsize=18)
        plt.ylabel(cfg["output_feature"], fontsize=18)
        plt.title(f'Sampling method: {cfg["sample_type"]}', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=15) # Adjust 14 to your desired size
        plt.savefig(os.path.join(output_file_path, cfg["sample_type"] + "_" + param_label + "_scatter.png"))
        plt.clf()
