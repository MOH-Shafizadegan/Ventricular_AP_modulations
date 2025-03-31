# import opencor as oc
import numpy as np
import random
import os
from Functions.OpenCor_Py.opencor_helper import SimulationHelper
import matplotlib
import matplotlib.pyplot as plt
from SALib.sample import latin
import time
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
from sklearn.linear_model import LinearRegression
import itertools
from sklearn.preprocessing import StandardScaler

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
output_file_path = "outputs/Single_param_analysis"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# sim specs
pre_time = 0
sim_time = 1000
dt = 1
n_mem_plot = 5

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
num_samples = 200
samples = latin.sample(problem, num_samples)

output_names = ['membrane/v']

# original AP
init_param_vals = sim_object.get_init_param_vals(param_names)
y, t = run_and_get_results(init_param_vals)
orig_v = np.squeeze(y)
orig_ADP90 = calculate_apd(t, orig_v, 90, depolarization_threshold=-20)


for i, param in enumerate(param_names):

    sim_object.set_param_vals(param_names, init_param_vals)

    mem_Vs = []
    APD90s = []
    for s in range(len(samples)):

        current_param_val = init_param_vals.copy()        
        current_param_val[i] = samples[s,i]

        y, t = run_and_get_results(current_param_val)
        v = np.squeeze(y)
        mem_Vs.append(v)
        APD90s.append(calculate_apd(t, v, 90, depolarization_threshold=-20))

        print(f"Param: {str(param)} - Iteration {s+1}/{len(samples)}.")

    match = re.search(r'(?<=/)(.*)', str(param))
    param_label = match.group(1)

    # plot the membrane action potentials
    selected_indices = random.sample(range(len(mem_Vs)), min(n_mem_plot, len(mem_Vs)))
    plt.plot(t, orig_v, linewidth=2, color='black', linestyle='--', label=f'Original AP - {param_label} = {init_param_vals[i]:.3f}')
    for j in selected_indices:
        plt.plot(t, mem_Vs[j], label=f'{param_label} = {samples[j,i]:.3f}')  # Assuming param_values holds corresponding values


    plt.xlabel('time (ms)')
    plt.ylabel('membrane potential (mV)')
    plt.grid(False)
    plt.legend()
    plt.savefig(output_file_path + "/" + param_label + "_AP.png")
    plt.clf()


    # Fit a line to output/input

    # normalize data
    x_min, x_max = np.min(np.squeeze(samples[:,i])), np.max(np.squeeze(samples[:,i]))
    x_norm = (np.squeeze(samples[:,i]) - x_min) / (x_max - x_min)

    y_min, y_max = np.min(APD90s), np.max(APD90s)
    y_norm = (APD90s - y_min) / (y_max - y_min)

    model = LinearRegression()
    x_reshape = x_norm.reshape(-1, 1)
    model.fit(x_reshape, y_norm)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(x_reshape)

    # scatter plot
    fig = plt.figure()
    init_x_norm = (init_param_vals[i] - x_min) / (x_max - x_min)
    init_y_norm = (orig_ADP90 - y_min) / (y_max - y_min)
    plt.scatter(init_x_norm, init_y_norm, c='black', marker='*')
    plt.scatter(x_norm, y_norm, c='r', marker='o')
    plt.plot(x_norm, y_pred, color='blue', label='Fitted line')
    equation_text = f"y = {slope:.2f}x + {intercept:.2f}"
    plt.text(0.95, 0.95, equation_text, transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, color='black')    
    plt.xlabel(param)
    plt.ylabel('ADP 90')
    plt.savefig(output_file_path + "/" + param_label + "_scatter.png")
    plt.clf()


