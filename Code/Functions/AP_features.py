import numpy as np

# def calculate_apd(time, voltage, repolarization_percentage, depolarization_threshold):
    
#     # Calculate the Action Potential Duration (APD) at a specified repolarization percentage.
    
    
#     if not (0 < repolarization_percentage < 100):
#         raise ValueError("Repolarization percentage must be between 0 and 100.")

#     # Identify Resting Membrane Potential (RMP) and Peak voltage
#     RMP = np.min(voltage)
#     V_peak = np.max(voltage)

#     # Calculate repolarization voltage level
#     V_repol = RMP + (1 - repolarization_percentage / 100) * (V_peak - RMP)
#     # print(V_repol)

#     # Find the index where depolarization starts (voltage crosses the depolarization threshold)
#     depolarization_indices = np.where(voltage > depolarization_threshold)[0]
#     if len(depolarization_indices) == 0:
#         return np.nan  # No valid APD
#     T_start_index = depolarization_indices[0]

#     # Find the index where voltage returns to V_repol during repolarization
#     repolarization_indices = np.where(voltage[T_start_index:] <= V_repol)[0]
#     if len(repolarization_indices) == 0:
#         return np.nan  # No valid APD
#     T_repol_index = T_start_index + repolarization_indices[0]

#     # Calculate APD
#     APD = time[T_repol_index] - time[T_start_index]

#     return APD

from scipy.interpolate import interp1d

def calculate_apd(time, voltage, repolarization_percentage, depolarization_threshold, resolution=0.001):
    """
    Calculate Action Potential Duration (APD) at a specified repolarization percentage,
    after interpolating the time-voltage signal to a finer resolution.
    
    Parameters:
        time (array): Time vector
        voltage (array): Voltage trace
        repolarization_percentage (float): Percentage to repolarization (e.g., 90 for APD90)
        depolarization_threshold (float): Voltage threshold for start of depolarization
        resolution (float): Desired time resolution for interpolation (default is 0.001)
    
    Returns:
        float: Calculated APD, or NaN if not found
    """

    if not (0 < repolarization_percentage < 100):
        raise ValueError("Repolarization percentage must be between 0 and 100.")

    # Interpolate time and voltage
    interp_func = interp1d(time, voltage, kind='cubic')
    fine_time = np.arange(time[0], time[-1], resolution)
    fine_voltage = interp_func(fine_time)

    # Identify Resting Membrane Potential (RMP) and Peak voltage
    RMP = np.min(fine_voltage)
    V_peak = np.max(fine_voltage)

    # Calculate repolarization voltage level
    V_repol = RMP + (1 - repolarization_percentage / 100) * (V_peak - RMP)

    # Find the index where depolarization starts (voltage crosses the depolarization threshold)
    depolarization_indices = np.where(fine_voltage > depolarization_threshold)[0]
    if len(depolarization_indices) == 0:
        return np.nan  # No valid APD
    T_start_index = depolarization_indices[0]

    # Find the index where voltage returns to V_repol during repolarization
    repolarization_indices = np.where(fine_voltage[T_start_index:] <= V_repol)[0]
    if len(repolarization_indices) == 0:
        return np.nan  # No valid APD
    T_repol_index = T_start_index + repolarization_indices[0]

    # Calculate APD
    APD = fine_time[T_repol_index] - fine_time[T_start_index]
    # print(round(APD, 6))

    return round(APD, 6)


def calculate_cycle_length(time, voltage, threshold=-30):
   
    # Calculate the cycle length between two sequential action potentials 
    
    # Find threshold crossings (upstroke phase)
    crossings = np.where((voltage[:-1] < threshold) & (voltage[1:] >= threshold))[0]

    if len(crossings) < 2:
        raise ValueError("Not enough action potentials detected to compute cycle length.")

    # Calculate cycle length between the first two threshold crossings
    cycle_length = time[crossings[1]] - time[crossings[0]]

    return cycle_length
