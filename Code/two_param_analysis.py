import numpy as np
import random
import os
from datetime import datetime
import pandas as pd
from Functions.OpenCor_Py.opencor_helper import SimulationHelper
import matplotlib
import matplotlib.pyplot as plt
from SALib.sample import latin
from scipy.stats import qmc
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
from sklearn.linear_model import LinearRegression
import itertools
from sklearn.preprocessing import MinMaxScaler

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

# Setting the output
output_type = 'APD90'
date_time = datetime.now().isoformat()
repolarization_level = int(re.search(r'\d+', output_type).group())

output_file_path = "outputs/two_param_analysis/" + date_time + "/" + output_type
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# sim specs
pre_time = 0
sim_time = 800
dt = 1
n_mem_plot = 5
sample_type = 'qmc'

is_plot_sample_AP = 1
is_plot_scatter = 0

# Create simulator object
sim_object = SimulationHelper(model_path, dt, sim_time, solver_info={'MaximumStep':0.001, 'MaximumNumberOfSteps':500000}, pre_time=pre_time)

# Define parametrs
param_names = ['extracellular/ko', 'IKr/GKr_b', 'ICaL/PCa_b']      # This list can be appended
param_vals_maxs = [5, 0.0121, 8.3757e-05]
param_vals_mins = [0.5 * val for val in param_vals_maxs]

# Latin hypercube sampling
problem = {
    'num_vars': len(param_names),
    'names': param_names,
    'bounds': list(zip(param_vals_mins, param_vals_maxs))
}
num_samples = 400
if sample_type == "lhs":
    samples = latin.sample(problem, num_samples)

elif sample_type == "qmc":
    sampler = qmc.Sobol(d=problem['num_vars'], scramble=True)  # Scrambled Sobol' for better uniformity
    qmc_samples = sampler.random(num_samples)
    samples = qmc.scale(qmc_samples, [b[0] for b in problem['bounds']], [b[1] for b in problem['bounds']])


output_names = ['membrane/v']

# original AP
init_param_vals = sim_object.get_init_param_vals(param_names)
y, t = run_and_get_results(init_param_vals)
orig_v = np.squeeze(y)
orig_APD90 = calculate_apd(t, orig_v, repolarization_level, depolarization_threshold=-20)


param_combinations = list(itertools.combinations(range(len(param_names)), 2))
for combo in param_combinations:

    param1, param2 = combo
    param_subset = [param_names[param1], param_names[param2]]
    
    print(f"\nRunning simulations for parameter combination: {param_subset}")
    
    # Reset parameters to initial values
    sim_object.set_param_vals(param_names, init_param_vals)

    mem_Vs = []
    APDs = []

    for s in range(len(samples)):
        current_param_val = init_param_vals.copy()        

        # Update only the selected parameters in this combination
        for idx in combo:
            current_param_val[idx] = samples[s, idx]

        # Run the simulation
        y, t = run_and_get_results(current_param_val)
        v = np.squeeze(y)

        # Store results
        mem_Vs.append(v)
        APDs.append(calculate_apd(t, v, repolarization_level, depolarization_threshold=-20))

        print(f"Params: {param_subset} - Iteration {s+1}/{len(samples)}.")


    # Save DataFrame with parameter values and APD
    param_labels = [re.search(r'(?<=/)(.*)', str(param)).group(1) if '/' in param else param for param in param_subset]
    param_label = "_".join(param_labels)

    df = pd.DataFrame(samples[:, combo], columns=param_labels)
    df[output_type] = APDs
    
    if not os.path.exists(f'{output_file_path}/dataset'):
        os.makedirs(f'{output_file_path}/dataset')
    df.to_csv(f"{output_file_path}/dataset/{param_label}_data.csv", index=False)

    # plot the membrane action potentials
    if is_plot_sample_AP:
        selected_indices = random.sample(range(len(mem_Vs)), min(n_mem_plot, len(mem_Vs)))
        fig = plt.figure(figsize=(10, 5))
        plt.plot(t, orig_v, linewidth= 2, color = 'black', linestyle='--', 
                label=f'Original AP \n {param_labels[0]} = {init_param_vals[param1]:.7f} | '
                                    f'{param_labels[1]} = {init_param_vals[param2]:.7f}')
        for j in selected_indices:
            plt.plot(t, mem_Vs[j], label=f'{param_labels[0]} = {samples[j,param1]:.7f} | '
                                        f'{param_labels[1]} = {samples[j,param2]:.7f}')

        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.grid(False)
        plt.legend()
        plt.title(f'Effects of {param_label}')
        plt.savefig(output_file_path + "/" + param_label + "_AP.png")
        plt.clf()

    # Fit a line to output/input
    if is_plot_scatter:
        # normalize data
        # scaler = MinMaxScaler(feature_range=(0, 1)) 
        # X = scaler.fit_transform(samples[:, combo])
        X = samples[:, combo] / np.max(samples[:, combo], axis=0)
        y = APDs


        model = LinearRegression()
        model.fit(X, y)
        coef1, coef2 = model.coef_
        intercept = model.intercept_


    # # scatter 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(init_param_vals[combo[0]], init_param_vals[combo[2]], orig_APD90, c='black', marker='*')
        ax.scatter(X[:, 0], X[:, 1], y, color='blue', label="Data")
        ax.set_xlabel(f"Normalized {param_subset[0]}")
        ax.set_ylabel(f"Normalized {param_subset[1]}")
        ax.set_zlabel(f"{output_type}")

        # Annotate with regression equation
        eq_text = f"{output_type} = {coef1:.6f} * {param_labels[0]} + {coef2:.6f} * {param_labels[1]} + {intercept:.3f}"
        ax.text2D(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        ax.set_title(f"Linear Regression: {param_labels[0]} and {param_labels[1]}")
        plt.savefig(output_file_path + "/" + f"{output_type}_{sample_type}_" + param_label + "_scatter3d.png")
        plt.clf()
