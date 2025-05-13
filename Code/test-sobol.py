# import numpy as np
# from SALib.sample import saltelli
# from SALib.analyze import sobol
# import matplotlib.pyplot as plt
# import os

# def simple_function(X):
#     """A simple quadratic function for testing Sobol analysis."""
#     x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
#     return x1**2 + 2*x2 + 3*x3**3

# output_file_path = "outputs/test"
# if not os.path.exists(output_file_path):
#     os.makedirs(output_file_path)


# # Define the problem
# problem = {
#     'num_vars': 3,
#     'names': ['x1', 'x2', 'x3'],
#     'bounds': [[-1, 1], [-1, 1], [-1, 1]]
# }

# # Generate samples
# param_values = saltelli.sample(problem, 1024, calc_second_order=True)

# # Evaluate the model
# Y = simple_function(param_values)

# # Perform Sobol sensitivity analysis
# Si = sobol.analyze(problem, Y)

# # Print results
# print("First-order indices:", Si['S1'])
# print("Total-order indices:", Si['ST'])


# # Plot Y against each parameter
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for i, ax in enumerate(axes):
#     ax.scatter(param_values[:, i], Y, alpha=0.5)
#     ax.set_xlabel(problem['names'][i])
#     ax.set_ylabel('Y')
#     ax.set_title(f'Y vs {problem["names"][i]}')
# plt.tight_layout()
# plt.savefig(output_file_path + "/" + "XY_scatter3d.png")
# plt.clf()

# import numpy as np
# import os
# from Functions.OpenCor_Py.opencor_helper import SimulationHelper
# import matplotlib
# import matplotlib.pyplot as plt
# from scipy.stats import qmc
# from SALib.analyze import sobol
# import SALib
# matplotlib.use('Agg')
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import re
# import seaborn as sns
# from SALib.sample import saltelli
# from Functions.AP_features import calculate_apd

# # Defining functions
# def run_and_get_results(param_vals):
#     sim_object.set_param_vals(param_names, param_vals)
#     sim_object.reset_states()
#     sim_object.run()
    
#     y = sim_object.get_results(output_names)
#     t = sim_object.tSim - pre_time
#     return y, t    

# # sim specs
# pre_time = 0
# sim_time = 1000
# dt = 1
# sample_type = 'qmc'

# working_dir = os.path.join(os.path.dirname(__file__))
# model_path = os.path.join(working_dir, "Models/ToRORd_dynCl_endo.cellml")
# output_names = ['membrane/v']

# # Create simulator object
# sim_object = SimulationHelper(model_path, dt, sim_time, solver_info={'MaximumStep':0.001, 'MaximumNumberOfSteps':500000}, pre_time=pre_time)

# param_names = ['extracellular/ko', 'IKr/GKr_b', 'IK1/GK1_b', 'ICaL/PCa_b']      # This list can be appended
# current_param_val = [5, 0.0121, 0.6992, 8.3757e-05]
# y, t = run_and_get_results(current_param_val)

# v = np.squeeze(y)
# calculate_apd(t, v, 50, depolarization_threshold=-20)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple

# # Load CSV file
# data = pd.read_csv('./outputs/two_param_analysis/ADP90/dataset/GKr_b_GK1_b_data.csv')

# columns = data.columns
# X = data.iloc[:, :-1].values  # First two columns as inputs
# Y = data.iloc[:, -1].values   # Last column as output

# # Compute standard deviation for each column
# std_values = {col: np.std(data[col]) for col in columns}
# print("Standard deviations:", std_values)

# # Fit a linear regression model
# model = LinearRegression()
# model.fit(X, Y)
# print("Linear Regression Coefficients:", model.coef_)
# print("Linear Regression Intercept:", model.intercept_)

# sigma_y = np.std(Y)
# idx_values = {columns[i]: (model.coef_[i] * std_values[columns[i]]) / sigma_y for i in range(len(model.coef_))}
# print("Index values:", idx_values)


from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


def separate_output_values(Y, D, N, calc_second_order):
    AB = np.zeros((N, D))
    BA = np.zeros((N, D)) if calc_second_order else None
    step = 2 * D + 2 if calc_second_order else D + 2

    A = Y[0 : Y.size : step]
    B = Y[(step - 1) : Y.size : step]
    for j in range(D):
        AB[:, j] = Y[(j + 1) : Y.size : step]
        if calc_second_order:
            BA[:, j] = Y[(j + 1 + D) : Y.size : step]

    return A, B, AB, BA


def extract_group_names(p: Dict) -> Tuple:
    """Get a unique set of the group names.

    Maintains specified order of group names.
    `groups` should be a list of parameter names if groups are not defined.

    Parameters
    ----------
    p : ProblemSpec or Dict

    Returns
    -------
    tuple : names, number of groups
    """
    if "groups" not in p or not p["groups"]:
        groups = p["names"]
    else:
        groups = p["groups"]

    names = list(pd.unique(groups))
    number = len(names)

    return names, number

# def first_order(A, AB, B):
#     """
#     First order estimator following Saltelli et al. 2010 CPC, normalized by
#     sample variance
#     """
#     y = np.r_[A, B]
#     if np.ptp(y) == 0:
#         warn(CONST_RESULT_MSG)
#         return np.array([0.0])

#     return np.mean(B * (AB - A), axis=0) / np.var(y, axis=0)


# calc_second_order = 1
# from scipy.stats import qmc

# problem = {
#     'num_vars': 3,
#     'names': ['x1', 'x2', 'x3'],
#     'bounds': [[0, 1], [0,10], [0,1]]
# }

# # param_values = saltelli.sample(problem, 1024)
# N = 1024 * 8
# sampler = qmc.Sobol(d=problem['num_vars'], scramble=True)  # Scrambled Sobol' for better uniformity
# qmc_samples = sampler.random(N)
# param_values = qmc.scale(qmc_samples, [b[0] for b in problem['bounds']], [b[1] for b in problem['bounds']])

# # print(param_values)

# # _, D = extract_group_names(problem)
# # print(D)
# Y = np.zeros([param_values.shape[0]])
# for i, X in enumerate(param_values):
#     # Y[i] = np.sum(X)
#     Y[i] = X[0] + X[1] + X[2]

# print(Y.size)

# if calc_second_order and Y.size % (2 * D + 2) == 0:
#         N = int(Y.size / (2 * D + 2))
#         print(N)
# elif not calc_second_order and Y.size % (D + 2) == 0:
#         N = int(Y.size / (D + 2))
# else:
#         raise RuntimeError(
#             """
#         Incorrect number of samples in model output file.
#         Confirm that calc_second_order matches option used during sampling."""
#         )
# print(N)

Y = (Y - Y.mean()) / Y.std()

A, B, AB, BA = separate_output_values(Y, D, N, calc_second_order)

# print(Y)
# print(A)
# print(B)
# print(B)
# Si = sobol.analyze(problem, Y)

# print(Si['S1'])


import numpy as np
from scipy.stats import qmc, sobol_indices

import scipy.stats
print(scipy.stats.sobol_indices)

# Define the problem
bounds = np.array([[0, 1], [0, 10], [0, 1]])
num_vars = bounds.shape[0]

# Generate Sobol samples
N = 2**13  # Must be a power of 2 for Sobol indices in scipy
sampler = qmc.Sobol(d=num_vars, scramble=True)
qmc_samples = sampler.random(N)

# Scale the samples to the problem bounds
param_values = qmc.scale(qmc_samples, bounds[:, 0], bounds[:, 1])

# Evaluate the model
Y = param_values[:, 0] + param_values[:, 1] + param_values[:, 2]

# Compute Sobol indices
S, ST = sobol_indices(Y, num_vars)

# Print the results
for i in range(num_vars):
    print(f"x{i+1} - First-order: {S[i]:.4f}, Total-order: {ST[i]:.4f}")


from scipy.stats import sobol_indices
import numpy as np

# Define the model
def model(X):
    return np.sum(X, axis=1)

# Run Sobol analysis
res = sobol_indices(
    func=model,
    n=2**13,
    dists=[(0, 1), (0, 10), (0, 1)],
    method='saltelli_2010',
)

# Extract and print indices
S = res['S1']
ST = res['ST']
for i, (s1, st) in enumerate(zip(S, ST)):
    print(f"x{i+1}: First-order = {s1:.4f}, Total-order = {st:.4f}")
