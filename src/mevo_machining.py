# -*- coding: utf-8 -*-
"""
MEVO: Meta-Evolutionary Optimizer with Surrogate-Based Optimization

This module implements an adaptive surrogate-based optimization algorithm that combines:
1. Gaussian Process (GP) or ANN surrogate models for function approximation
2. MEPSO (Micro Evolutionary Particle Swarm Optimizer) for acquisition function optimization
3. Evolutionary operators (crossover and mutation) for generating new candidate solutions
4. Active learning strategies with adaptive search space reduction

The algorithm iteratively:
- Samples new points by optimizing an acquisition function
- Updates the surrogate model with actual function evaluations
- Reduces the search space based on elite solutions
- Maintains an archive of evaluated solutions to avoid redundant evaluations

Authors: Antonio Velazquez-Lopez, Maximiliano Aceves, and Rafael Batres
Institution: Tecnologico de Monterrey
Date: June 2, 2026

References:
    Velázquez-López, A., Batres, R., Miranda-Valenzuela, J. C., & Calderón-Nájera, J. D. (2026). 
    Experimental assessment of the metamodel-based evolutionary optimizer (MEVO) for machining optimization. 
    International Journal of Advanced Manufacturing Technology
"""

####################################################################################
# IMPORTS
####################################################################################


# The following libraries are required:
# pandas, matplotlib, scipy, sklearn

from __future__ import annotations
import subprocess
import pandas as pd
from matplotlib import pyplot as plt
import shutil
import fileinput
from time import process_time, time
from csv import reader
import os
import scipy.stats as st
import random
import sys
import copy
import csv
import numpy as np
import math
from math import sqrt, exp, pi
from numpy import vstack, argmin, asarray, array
from numpy.random import uniform, random
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as ConstantKer
from warnings import catch_warnings, simplefilter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from mepso import *  # Assumes mepso.py is in the same directory and contains the MEPSO implementation
from datetime import datetime
import pickle  # For saving and loading algorithm state

# --- FLEXIBLE GOOGLE COLAB SUPPORT ---
# Attempt to detect and mount Google Drive if running on Colab
IS_COLAB = False
DRIVE_PATH = ""

try:
    from google.colab import drive
    IS_COLAB = True
    try:
        drive.mount('/content/drive')
        DRIVE_PATH = "/content/drive/MyDrive/"
        print("[INFO] Google Drive mounted successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to mount Google Drive: {e}")
        DRIVE_PATH = ""  # Use local storage if mounting fails
except ImportError:
    # Not running on Colab; proceed with local storage
    IS_COLAB = False
    DRIVE_PATH = ""

# Define checkpoint folder and file paths
CHECKPOINT_FOLDER = f"{DRIVE_PATH}MEVO/checkpoint/" if DRIVE_PATH else "./checkpoint/"
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
CHECKPOINT_FILE = f"{CHECKPOINT_FOLDER}mevo_checkpoint.pkl"


####################################################################################
# CHECKPOINT FUNCTIONS
####################################################################################

def save_checkpoint(state, filename=None):
    """
    Save the complete state of the MEVO algorithm to a file for resuming interrupted runs.
    
    This function serializes all necessary algorithm state including the surrogate model,
    training data, iteration counters, and random state to enable resuming optimization
    from where it was interrupted.
    
    Parameters
    ----------
    state : dict
        Dictionary containing the algorithm state with keys:
        - 'Xsamples': array of evaluated decision variable vectors
        - 'yactual': list of objective function values
        - 'all_results_log': complete evaluation history
        - 'model': trained surrogate model object
        - 'scaler_x': MinMaxScaler fitted on input variables
        - 'scaler_y': MinMaxScaler fitted on objective values
        - 'iteration_num': current iteration number
        - 'evaluated_points_count': total evaluations performed
        - 'best_y_tracker': best objective value found so far
        - 'convergence_data_plot': convergence history
        - 'num_initial_points': number of initial LHS points
        - 'random_state': Python random module state (optional)
        - 'np_random_state': NumPy random module state (optional)
    
    filename : str, optional
        Path to save the checkpoint file. If None, uses CHECKPOINT_FILE global variable.
    
    Returns
    -------
    None
    
    Notes
    -----
    The checkpoint file is created in pickle format for Python serialization compatibility.
    Directories are created automatically if they do not exist.
    """
    if filename is None:
        filename = CHECKPOINT_FILE

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except Exception as e:
        print(f"[WARNING] Could not create checkpoint directory {os.path.dirname(filename)}: {e}")

    try:
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"[INFO] Checkpoint saved successfully to: {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")

def load_checkpoint(filename=None):
    """
    Load a previously saved algorithm state from a checkpoint file.
    
    This function deserializes the algorithm state to resume an interrupted optimization run.
    If the checkpoint file does not exist or cannot be loaded, returns None to start fresh.
    
    Parameters
    ----------
    filename : str, optional
        Path to the checkpoint file to load. If None, uses CHECKPOINT_FILE global variable.
    
    Returns
    -------
    dict or None
        Deserialized state dictionary if successful (see save_checkpoint for keys).
        Returns None if file does not exist or loading fails.
    
    Notes
    -----
    If checkpoint loading fails, the algorithm will start a new run with initial LHS points.
    Any exceptions during loading are caught and reported to allow graceful fallback.
    """
    if filename is None:
        filename = CHECKPOINT_FILE
    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        print(f"[INFO] Checkpoint loaded successfully from: {filename}")
        return state
    except FileNotFoundError:
        print(f"[INFO] No checkpoint found at: {filename}. Starting new run.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return None

####################################################################################
# ALGORITHM CONFIGURATION PARAMETERS
####################################################################################

# Random seed for reproducible results
# Set to a fixed integer for reproducible runs, or None for non-deterministic behavior
seed = 42
random.seed(seed)
np.random.seed(seed)

# ============================================================================
# 1. EVALUATION MODE
# ============================================================================
# Specifies how objective function values (SE, SR) are obtained.
# - 'manual': Program prompts user to input observed values after each evaluation
# - 'automatic': Program uses fitted regression models to predict values
#
# Use 'manual' when performing physical experiments or simulations.
# Use 'automatic' when validating algorithm with surrogate models.
EVALUATION_MODE = 'manual'

# ============================================================================
# 2. OPTIMIZATION MODE
# ============================================================================
# Specifies the overall optimization strategy:
# - 'single_objective': Optimize a single objective variable directly
# - 'DF_SOO': Use desirability functions for single-objective (Desirability Function)
# - 'DF_MOO': Use desirability functions for multi-objective optimization
#
# DF_MOO allows simultaneous optimization of multiple conflicting objectives
# by combining them into a composite desirability index.
OPTIMIZATION_MODE = "DF_MOO"

# ============================================================================
# 3. OPTIMIZATION GOAL
# ============================================================================
# Specifies the primary objective variable when using 'single_objective' or 'DF_SOO'.
# Valid options: "SE" (Surface Finish), "SR" (Stress), "MRR" (Material Removal Rate)
# 
# Only relevant if OPTIMIZATION_MODE is 'single_objective' or 'DF_SOO'.
# Ignored when OPTIMIZATION_MODE is 'DF_MOO'.
OPTIMIZATION_GOAL = "SR"

# ============================================================================
# 4. MULTI-OBJECTIVE VARIABLES (DF-MOO)
# ============================================================================
# Specifies which variables to include in multi-objective optimization.
# Only used when OPTIMIZATION_MODE is 'DF_MOO'.
#
# Example: ["SE", "SR", "MRR"] optimizes all three variables
#          ["SE", "SR"] optimizes only surface finish and stress
DF_MOO_VARIABLES = ["SE", "SR"]

# ============================================================================
# 5. OPTIMIZATION INTENTIONS
# ============================================================================
# Defines whether each variable should be minimized or maximized.
# Used by the desirability function calculations.
#
# For machining operations:
# - SE (Surface Roughness): "min" (lower is better)
# - SR (Stress): "min" (lower is better)
# - MRR (Material Removal Rate): "max" (higher is better)
OPTIMIZATION_INTENTIONS = {
    "SE": "min",
    "SR": "min",
    "MRR": "max"
}

# ============================================================================
# 6. DESIRABILITY RANGES
# ============================================================================
# Defines the range for desirability function calculations for each variable.
# Typically derived from Design of Experiments (DoE) or literature values.
#
# For minimization (min): (ideal_value, worst_value)
#   - ideal_value: minimum acceptable/observed value (d=1.0)
#   - worst_value: maximum acceptable/observed value (d=0.0)
#
# For maximization (max): (worst_value, ideal_value)
#   - worst_value: minimum acceptable/observed value (d=0.0)
#   - ideal_value: maximum acceptable/observed value (d=1.0)
#
# Desirability function maps values outside these bounds based on TRUNCATE_DESIRABILITY setting.
DESIRABILITY_RANGES = {
    "SE": (5.91011, 16.6839),           # (ideal, worst) for minimization
    "SR": (0.66150, 3.0100),            # (ideal, worst) for minimization
    "MRR": (3850.276383, 12994.682794)  # (worst, ideal) for maximization
}

# ============================================================================
# 7. DESIRABILITY WEIGHTS
# ============================================================================
# Relative importance (weights) for each objective in multi-objective optimization.
# Higher weights increase the influence of that objective on the composite desirability.
#
# The composite desirability is calculated as the weighted geometric mean:
# d_total = (d_SE^w_SE * d_SR^w_SR * d_MRR^w_MRR)^(1 / sum(weights))
#
# For equal importance, use equal weights (e.g., 1, 1, 1).
# To prioritize one objective, increase its weight (e.g., SE: 2, SR: 1, MRR: 1).
WEIGHTS = {"SE": 1, "SR": 1, "MRR": 1}

# ============================================================================
# 8. STOPPING CRITERION AND EVOLUTIONARY OPERATORS
# ============================================================================
# Maximum number of objective function evaluations before stopping optimization
MAX_EVALUATIONS = 16

# Probability of applying crossover operator in each iteration
# Valid range: [0.0, 1.0]
# Higher values increase exploration through recombination of elite solutions
PROB_CROSSOVER = 0.1

# Probability of applying mutation operator in each iteration
# Valid range: [0.0, 1.0]
# Higher values increase exploration through perturbation around best solution
PROB_MUTATION = 0.2

# ============================================================================
# 9. DESIRABILITY TRUNCATION MODE
# ============================================================================
# Controls how desirability values outside the specified ranges are handled:
# - True: Clip desirability values to [0.0, 1.0] range
# - False: Allow extrapolation (values can exceed bounds)
#
# Typically set to True for robust behavior with unexpected input values.
TRUNCATE_DESIRABILITY = True

# ============================================================================
# 10. INITIAL DATA FILE
# ============================================================================
# CSV file containing initial Latin Hypercube Sampling (LHS) points
# Expected format: CSV with columns [index, ap, fz, Vc, SE, SR]
# where ap=depth of cut, fz=feed per tooth, Vc=cutting speed
CSV_FILE_NAME = 'LHS_MEVO_4_Points.csv'


####################################################################################
# DESIRABILITY FUNCTION AND UTILITY FUNCTIONS
####################################################################################

def calculate_desirability(value, range_min, range_max, goal='min'):
    """
    Calculate a desirability index for an individual response variable.
    
    The desirability function maps variable values to a [0, 1] scale where:
    - d = 1.0 represents the ideal value
    - d = 0.0 represents the worst acceptable value
    - d = 0.5 represents the midpoint
    
    Uses linear interpolation within the specified range and respects the
    TRUNCATE_DESIRABILITY global parameter for values outside the range.
    
    Parameters
    ----------
    value : float
        The response variable value to evaluate.
    
    range_min : float
        Minimum of the desirability range.
    
    range_max : float
        Maximum of the desirability range.
    
    goal : str
        Optimization direction:
        - 'min': Minimize the response (lower values are better)
        - 'max': Maximize the response (higher values are better)
    
    Returns
    -------
    float
        Desirability value in [0.0, 1.0] if TRUNCATE_DESIRABILITY is True,
        or potentially outside this range if TRUNCATE_DESIRABILITY is False
        (allowing extrapolation for values beyond the specified range).
    
    Notes
    -----
    If range_min == range_max, returns 1.0 if value is at the ideal point,
    0.0 otherwise.
    
    For minimization: ideal at range_min, worst at range_max
    For maximization: worst at range_min, ideal at range_max
    """

    if goal == 'min':
        ideal, worst = range_min, range_max
        if ideal == worst:
            result = 1.0 if value <= ideal else 0.0
        else:
            result = (worst - value) / (worst - ideal)


        if TRUNCATE_DESIRABILITY:
            result = max(0.0, min(1.0, result))


        return result

    else: # goal == 'max'
        worst, ideal = range_min, range_max
        if ideal == worst:
            result = 1.0 if value >= ideal else 0.0
        else:
            result = (value - worst) / (ideal - worst)


        if TRUNCATE_DESIRABILITY:
            result = max(0.0, min(1.0, result))


        return result

def calculate_mrr(x_point):
    """
    Calculate Material Removal Rate (MRR) from machining parameters.
    
    The MRR is calculated using the standard formula:
    MRR = D * ap * (fz * (1000*Vc)/(pi*D) * z)
    
    where D is the tool diameter and z is the number of flutes.
    
    Parameters
    ----------
    x_point : array-like
        Decision variable vector containing:
        - x_point[0] = ap (depth of cut) [mm]
        - x_point[1] = fz (feed per tooth) [mm/tooth]
        - x_point[2] = Vc (cutting speed) [m/min]
    
    Returns
    -------
    float
        Material Removal Rate [mm^3/min]
    
    Notes
    -----
    Hard-coded parameters for the current tool configuration:
    - Cutter diameter (D) = 6.35 mm
    - Number of flutes (z) = 4
    """
    D = 6.35  # Cutter diameter [mm]
    z = 4     # Number of flutes
    ap, fz, Vc = x_point[0], x_point[1], x_point[2]
    MRR = D * ap * (fz * ((1000 * Vc) / (pi * D)) * z)
    return MRR

def predict_SE_automatico(x_point):
    """
    Predict Surface Finish (SE) using a fitted quadratic regression model.
    
    This function implements a second-order response surface model derived from
    Design of Experiments (DoE) for predicting surface roughness (SE) based on
    machining parameters.
    
    Parameters
    ----------
    x_point : array-like
        Decision variable vector containing:
        - x_point[0] = ap (depth of cut) [mm]
        - x_point[1] = fz (feed per tooth) [mm/tooth]
        - x_point[2] = Vc (cutting speed) [m/min]
    
    Returns
    -------
    float
        Predicted surface finish value [microns or relevant unit]
    
    Notes
    -----
    This is a fitted response surface model obtained from experimental data.
    The model includes linear, quadratic, and interaction terms.
    """
    ap, fz, Vc = x_point[0], x_point[1], x_point[2]
    SE = (90.894
          - 30.501 * ap
          - 851.2 * fz
          - 0.62141 * Vc
          + 4.216 * ap**2
          + 3275.1 * fz**2
          + 0.001871 * Vc**2
          + 117.77 * ap * fz
          + 0.07743 * ap * Vc
          + 2.0918 * fz * Vc)
    return SE

def predict_SR_automatico(x_point):
    """
    Predict Stress (SR) using a fitted quadratic regression model.
    
    This function implements a second-order response surface model derived from
    Design of Experiments (DoE) for predicting tool stress (SR) based on
    machining parameters.
    
    Parameters
    ----------
    x_point : array-like
        Decision variable vector containing:
        - x_point[0] = ap (depth of cut) [mm]
        - x_point[1] = fz (feed per tooth) [mm/tooth]
        - x_point[2] = Vc (cutting speed) [m/min]
    
    Returns
    -------
    float
        Predicted stress value [MPa or relevant unit]
    
    Notes
    -----
    This is a fitted response surface model obtained from experimental data.
    The model includes linear, quadratic, and interaction terms.
    """
    ap, fz, Vc = x_point[0], x_point[1], x_point[2]
    SR = (-3.098
          + 2.595 * ap
          + 92.6 * fz
          - 0.00829 * Vc
          - 0.767 * ap**2
          - 578 * fz**2)
    return SR




def evaluate_new_point(x_point):
    """
    Evaluate a candidate solution and compute its objective value.
    
    This function performs the following steps:
    1. Obtain SE and SR values (manually or via prediction models)
    2. Calculate MRR deterministically from machining parameters
    3. Compute individual desirability indices for each response
    4. Calculate composite multi-objective desirability if enabled
    5. Convert desirability to optimization objective (negated for maximization)
    
    Parameters
    ----------
    x_point : array-like
        Decision variable vector [ap, fz, Vc]
    
    Returns
    -------
    dict
        Evaluation results containing:
        - 'actual': objective value for optimizer (negative for maximization)
        - 'se': surface finish value
        - 'sr': stress value
        - 'mrr': material removal rate
        - 'df_se': desirability of SE
        - 'df_sr': desirability of SR
        - 'df_mrr': desirability of MRR
        - 'df_moo': composite multi-objective desirability
    
    Notes
    -----
    The 'actual' value is negated so that minimizing it is equivalent to
    maximizing the underlying desirability function (for optimizer compatibility).
    """
    # Step 1: Obtain SE and SR values
    if EVALUATION_MODE == 'automatic':
        se_val = predict_SE_automatico(x_point)
        sr_val = predict_SR_automatico(x_point)
    else:
        print("-" * 50)
        print(f"Evaluating new point manually: {np.round(x_point, 5)}")
        se_val = float(input("--> Enter SE value: "))
        sr_val = float(input("--> Enter SR value: "))
        print("-" * 50)

    mrr_val = calculate_mrr(x_point)
    raw_values = {"SE": se_val, "SR": sr_val, "MRR": mrr_val}

    # Step 2: Calculate individual desirabilities
    desirabilities = {}
    for var, (range_min, range_max) in DESIRABILITY_RANGES.items():
        goal = OPTIMIZATION_INTENTIONS[var]
        desirabilities[var] = calculate_desirability(raw_values[var], range_min, range_max, goal)

    # Step 3: Calculate composite multi-objective desirability (DF-MOO)
    d_prod = 1.0
    weight_sum = 0
    active_vars = DF_MOO_VARIABLES
    for var in active_vars:
        base = desirabilities[var]
        if base < 0:
            base = 0
        d_prod *= (base ** WEIGHTS[var])
        weight_sum += WEIGHTS[var]

    df_moo = d_prod ** (1 / weight_sum) if weight_sum > 0 else 0.0

    # Step 4: Select objective value for the optimizer
    if OPTIMIZATION_MODE == "single_objective":
        actual_value = raw_values[OPTIMIZATION_GOAL]
        if OPTIMIZATION_INTENTIONS[OPTIMIZATION_GOAL] == 'max':
            actual_value *= -1
    elif OPTIMIZATION_MODE == "DF_SOO":
        actual_value = -desirabilities[OPTIMIZATION_GOAL]  # Maximize desirability
    elif OPTIMIZATION_MODE == "DF_MOO":
        actual_value = -df_moo  # Maximize composite desirability
    else:
        raise ValueError(f"Invalid OPTIMIZATION_MODE value: '{OPTIMIZATION_MODE}'")

    # Step 5: Return complete evaluation results
    results = {
        'actual': actual_value, 'se': se_val, 'sr': sr_val, 'mrr': mrr_val,
        'df_se': desirabilities['SE'], 'df_sr': desirabilities['SR'], 'df_mrr': desirabilities['MRR'],
        'df_moo': df_moo
    }
    return results


def surrogate(model, x1, x2, x3):
    """
    Predict objective function value using the fitted surrogate model.
    
    This function evaluates the surrogate model (ANN, Random Forest, or SVM)
    at a given point. It applies the same scaling transformations used during
    model training to ensure consistent predictions.
    
    Parameters
    ----------
    model : sklearn estimator
        Fitted surrogate model (MLPRegressor, RandomForestRegressor, or SVR).
    
    x1, x2, x3 : float
        Decision variables (ap, fz, Vc)
    
    Returns
    -------
    tuple of (float, float)
        - Predicted objective value (scaled back to original range)
        - Standard deviation (0.0 placeholder for compatibility)
    
    Notes
    -----
    Uses global scaler_x and scaler_y for input/output transformation.
    Warnings are suppressed to avoid convergence warnings from sklearn.
    """
    global scaler_x, scaler_y
    Sol = [[x1, x2, x3]]
    with catch_warnings():
        simplefilter("ignore")
        rescaledX = scaler_x.transform(Sol)
        surr_prediction = model.predict(rescaledX)
        surr_prediction = scaler_y.inverse_transform(surr_prediction.reshape(1, -1))
        return surr_prediction[0][0], 0  # Returns value and dummy std dev

def acquisition(x1, x2, x3):
    """
    Acquisition function for guiding the PSO search.
    
    Wraps the surrogate model to provide a continuous function for
    optimization by the MicroEPSO algorithm.
    
    Parameters
    ----------
    x1, x2, x3 : float
        Decision variables (ap, fz, Vc)
    
    Returns
    -------
    float
        Predicted objective value from the surrogate model
    """
    yhat, _ = surrogate(model, x1, x2, x3)
    return yhat

def opt_acquisition_3d(X, model):
    """
    Optimize the acquisition function using Micro Evolutionary PSO.
    
    Uses the MicroEPSO algorithm to search for the point most likely to
    improve the objective function based on the surrogate model. This
    balances exploration and exploitation in the decision space.
    
    Parameters
    ----------
    X : array
        Current sample points (used for context, not directly in optimization)
    
    model : sklearn estimator
        Fitted surrogate model
    
    Returns
    -------
    array
        Best point found by the PSO optimizer [x1, x2, x3]
    
    Notes
    -----
    Uses global lower_bound and upper_bound for search space constraints.
    MicroEPSO hyperparameters:
    - iterations: Inner loop iterations (higher = more local refinement)
    - max_epochs: Outer loop epochs (higher = more outer iterations)
    - beta: Probability of following global best (0.9 = mostly global)
    - alfa: Probability of following local best (0.6 = some local)
    - mu: Probability of mutation (0.5 = 50% chance)
    - sigma: Mutation standard deviation (0.7 = moderate perturbation)
    - gamma: Crossover weight (0.7 = 70% from first parent)
    """
    global lower_bound, upper_bound
    pso = MicroEPSO(acquisition, (lower_bound, upper_bound),
                     iterations=15,      # Inner loop iterations
                     max_epochs=2,       # Outer loop epochs
                     population_size=20, # Population size
                     beta=0.9,           # Probability for global best movement
                     alfa=0.6,           # Probability for local best movement
                     mu=0.5,             # Mutation probability
                     sigma=0.7,          # Mutation standard deviation
                     gamma=0.7)          # Crossover weight
    pso.run()
    return pso.global_best.best_particle




####################################################################################
# MAIN OPTIMIZATION SCRIPT
####################################################################################

# Decision variable bounds for machining parameters
upper_bound = [1.8, 0.063, 90]  # [depth of cut, feed per tooth, cutting speed]
lower_bound = [1.0, 0.035, 50]

# Number of independent optimization runs
# Note: runs are separate optimization cycles, distinct from iterations within a run
number_of_runs = 1
best_solution_array, best_cost_array, comp_time_array, act_time_array = [], [], [], []

# Main loop for multiple optimization runs
for run_num in range(number_of_runs):
    print(f"\n--- Starting Run {run_num + 1}/{number_of_runs} ---")

    # Attempt to load checkpoint for resuming interrupted runs
    checkpoint = load_checkpoint()

    if checkpoint:
        # Restore all algorithm state variables from checkpoint
        Xsamples = checkpoint['Xsamples']
        yactual = checkpoint['yactual']
        all_results_log = checkpoint['all_results_log']
        model = checkpoint['model']
        scaler_x = checkpoint['scaler_x']
        scaler_y = checkpoint['scaler_y']
        iteration_num = checkpoint['iteration_num']
        evaluated_points_count = checkpoint['evaluated_points_count']
        best_y_tracker = checkpoint['best_y_tracker']
        convergence_data_plot = checkpoint['convergence_data_plot']
        num_initial_points = checkpoint['num_initial_points']

        # Restore random number generator state for reproducibility
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['np_random_state'])
            print("[INFO] Random state restored from checkpoint.")

        start_act_time = time()
        start_time = process_time()

    else:
        # Start new optimization run with initial Latin Hypercube Sampling (LHS) points
        start_act_time = time()
        start_time = process_time()

        # Dictionary to store all evaluation results (SE, SR, MRR) for each point
        all_results_log = []

        # Load initial LHS design points from CSV file
        df = pd.read_csv(CSV_FILE_NAME)


        yactual = []
        initial_points = []
        for i in range(len(df)):
            point = [df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3]]
            initial_points.append(point)

            # Extract observed response values from CSV
            se_val = df.iloc[i, 4]
            sr_val = df.iloc[i, 5]
            mrr_val = calculate_mrr(point)
            raw_values = {"SE": se_val, "SR": sr_val, "MRR": mrr_val}

            # Calculate individual desirabilities for each response variable
            desirabilities = {}
            for var, (range_min, range_max) in DESIRABILITY_RANGES.items():
                goal = OPTIMIZATION_INTENTIONS[var]
                desirabilities[var] = calculate_desirability(raw_values[var], range_min, range_max, goal)

            # Calculate composite multi-objective desirability
            d_prod = 1.0
            weight_sum = 0
            for var in DF_MOO_VARIABLES:
                if var in desirabilities and var in WEIGHTS:
                    base = desirabilities[var]
                    if base < 0:
                        base = 0
                    d_prod *= (base ** WEIGHTS[var])
                    weight_sum += WEIGHTS[var]
            df_moo = d_prod ** (1 / weight_sum) if weight_sum > 0 and d_prod >= 0 else 0.0

            # Determine objective value for the optimizer
            if OPTIMIZATION_MODE == "single_objective":
                actual_value = raw_values[OPTIMIZATION_GOAL]
                if OPTIMIZATION_INTENTIONS[OPTIMIZATION_GOAL] == 'max':
                    actual_value *= -1
            elif OPTIMIZATION_MODE == "DF_SOO":
                actual_value = -desirabilities[OPTIMIZATION_GOAL]
            elif OPTIMIZATION_MODE == "DF_MOO":
                actual_value = -df_moo
            else:
                raise ValueError(f"Invalid OPTIMIZATION_MODE value: '{OPTIMIZATION_MODE}'")

            # Store complete evaluation results
            results = {
                'actual': actual_value, 'se': se_val, 'sr': sr_val, 'mrr': mrr_val,
                'df_se': desirabilities['SE'], 'df_sr': desirabilities['SR'], 'df_mrr': desirabilities['MRR'],
                'df_moo': df_moo
            }
            all_results_log.append(results)
            yactual.append(results['actual'])

        Xsamples = array(initial_points)
        num_initial_points = len(initial_points)

        # Initialize and train surrogate model (Artificial Neural Network)
        # Alternative models available:
        # - RandomForestRegressor(n_estimators=100, random_state=None)
        # - SVR(kernel='rbf', C=1, epsilon=0.05, gamma='scale')
        model = MLPRegressor(
            hidden_layer_sizes=(8,),      # Number of neurons in hidden layer
            activation='logistic',         # Activation function (logistic, tanh, relu)
            solver='lbfgs',               # Optimization algorithm (lbfgs, adam)
            max_iter=500,
            learning_rate='adaptive',
            learning_rate_init=0.01,
            alpha=0.0001,                # L2 regularization
            random_state=None
        )

        # Scale input and output variables to [0, 1] range for model training
        scaler_x = MinMaxScaler().fit(Xsamples)
        rescaledX = scaler_x.transform(Xsamples)
        yactual_arr = asarray(yactual).reshape(-1, 1)
        scaler_y = MinMaxScaler().fit(yactual_arr)
        rescaledY = scaler_y.transform(yactual_arr)
        model.fit(rescaledX, rescaledY.ravel())

        # Evaluate initial model quality on LHS points
        y_pred_init = model.predict(rescaledX)
        r2_init = r2_score(rescaledY, y_pred_init)
        rmse_init = sqrt(mean_squared_error(rescaledY, y_pred_init))

        print(f"\n[INFO] Initial surrogate model trained. R-squared: {r2_init:.4f} | RMSE: {rmse_init:.5f}")

        # Initialize counters for new optimization run
        iteration_num = 0
        evaluated_points_count = 0
        best_y_tracker = float('inf')
        convergence_data_plot = []


    # Create output files for results reporting
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    stats_file_name = f"statistics_{date_time}_run{run_num+1}.csv"
    stats_file = open(stats_file_name, 'w', encoding='UTF8', newline='')
    stats_writer = csv.writer(stats_file)
    header = ['iter', 'type', 'x1', 'x2', 'x3', 'obj_value', 'best_obj_so_far',
              'SE', 'SR', 'MRR', 'DF_SE', 'DF_SR', 'DF_MRR', 'DF_MOO', 'rmse', 'r2']
    stats_writer.writerow(header)

    convergence_file_name = f"convergence_{date_time}_run{run_num+1}.csv"
    convergence_file = open(convergence_file_name, 'w', encoding='UTF8', newline='')
    convergence_writer = csv.writer(convergence_file)
    convergence_header = ['iter', 'best_x1', 'best_x2', 'best_x3', 'best_obj_value']
    convergence_writer.writerow(convergence_header)

    # Initialize iteration tracking variables
    points_found_in_previous_iteration = 0
    total_points_in_model = len(Xsamples)
    r_squared = 'N/A'

    # Las variables 'evaluated_points_count', 'iteration_num', 'best_y_tracker'
    # y 'convergence_data_plot' YA TIENEN LOS VALORES CORRECTOS (cargados o nuevos).

    # Main optimization loop
    while evaluated_points_count < MAX_EVALUATIONS:
        iteration_num += 1

        print(f"\n{'='*20} Iteration {iteration_num} (Evaluations: {evaluated_points_count}/{MAX_EVALUATIONS}) {'='*20}")

        # Report iteration progress
        if iteration_num > 1:
            print(f"[INFO] Evaluated {points_found_in_previous_iteration} point(s) from previous iteration.")
            print(f"[INFO] Surrogate model updated. Total points in model: {total_points_in_model} (including {num_initial_points} initial LHS points)")

        points_found_this_iteration = 0

        print("--- 1. Main Point Search (PSO Acquisition Optimization) ---")

        # Optimize the acquisition function to find the most promising point
        x = opt_acquisition_3d(Xsamples, model)

        # Check for duplicate (already evaluated point)
        x_copy = np.round(x, 12)
        Xsamples_list = np.round(Xsamples, 12).tolist()

        point_is_new = True
        if x_copy.tolist() in Xsamples_list:
            print(f"[INFO] Duplicate point detected: {x_copy.tolist()}. Using cached result.")
            point_is_new = False
            results = all_results_log[Xsamples_list.index(x_copy.tolist())]
        else:
            # Evaluate the new point and update data
            results = evaluate_new_point(x)
            evaluated_points_count += 1
            points_found_this_iteration += 1
            all_results_log.append(results)
            Xsamples = vstack((Xsamples, x))
            yactual.append(results['actual'])

            current_best = min(yactual)
            convergence_data_plot.append(current_best)


        # Find and report best result so far
        ix = argmin(yactual)
        best_x_so_far = Xsamples[ix]
        best_y_so_far = yactual[ix]

        # Calculate model evaluation metrics
        y_pred = model.predict(scaler_x.transform(Xsamples))
        rescaledY_actual = scaler_y.transform(asarray(yactual).reshape(-1, 1))
        rmse = sqrt(mean_squared_error(rescaledY_actual, y_pred))
        r_squared = r2_score(rescaledY_actual, y_pred)

        # Report PSO point
        print(f"PSO suggested point: {np.round(x, 5)} -> SE: {results['se']:.5f}, SR: {results['sr']:.5f} -> Objective: {results['actual']:.5f}")
        ix = argmin(yactual)
        best_y_so_far = yactual[ix]
        stats_data = [iteration_num, 'pso', x[0], x[1], x[2], results['actual'], best_y_so_far,
                      results['se'], results['sr'], results['mrr'],
                      results['df_se'], results['df_sr'], results['df_mrr'], results['df_moo'], rmse, r_squared]
        stats_writer.writerow(stats_data)

        convergence_data = [iteration_num, best_x_so_far[0], best_x_so_far[1], best_x_so_far[2], best_y_so_far]
        convergence_writer.writerow(convergence_data)

        # Evolutionary Operators (optional, probabilistic)
        # Crossover operator
        if uniform() < PROB_CROSSOVER and evaluated_points_count < MAX_EVALUATIONS:
            print("\n--- 2. Optional Point (Crossover Operator) ---")
            # Selección de padres elitista
            best_n_sols = []
            ordered_y_indices = np.argsort(yactual)
            best_n = min(5, len(yactual)) # Tomar los 5 mejores o menos si no hay suficientes
            for j in range(best_n):
                best_n_sols.append(Xsamples[ordered_y_indices[j]])

            # Elegir padres distintos del grupo de élite
            k = 0
            while True:
                chosen_dad_idx = random.randint(0, best_n - 1)
                dad = best_n_sols[chosen_dad_idx]
                chosen_mom_idx = random.randint(0, best_n - 1)
                mom = best_n_sols[chosen_mom_idx]
                if list(dad) != list(mom) or k > best_n: break # Evitar bucle infinito
                k += 1

            # Aplicación del operador de cruce (mezcla por dimensión)
            gamma = 0.7
            alpha = [random.uniform(-gamma, 1 + gamma) for _ in range(len(dad))]
            son = [alpha[j] * dad[j] + (1 - alpha[j]) * mom[j] for j in range(len(dad))]
            son = np.clip(son, lower_bound, upper_bound)

            # Check for duplicate crossover offspring
            Xsamples_list = np.round(Xsamples, 12).tolist()

            if son_copy.tolist() in Xsamples_list:
                print(f"Punto de crossover repetido detectado...")
            else: # Evaluar, actualizar y registrar el nuevo punto 'son'
                results_son = evaluate_new_point(son)
                # Imprime el punto, los valores de entrada (SE/SR) y el objetivo
                print(f" 	-> Punto Crossover generado: {np.round(son, 5)} -> SE: {results_son['se']:.5f}, SR: {results_son['sr']:.5f} -> Resultado (objetivo): {results_son['actual']:.5f}")
                evaluated_points_count += 1
                points_found_this_iteration += 1
                all_results_log.append(results_son)
                Xsamples = vstack((Xsamples, son))
                yactual.append(results_son['actual'])

                current_best = min(yactual)
                convergence_data_plot.append(current_best)

                ix = argmin(yactual)
                best_y_so_far = yactual[ix]
                stats_data = [iteration_num, 'crossover', son[0], son[1], son[2], results_son['actual'], best_y_so_far,
                              results_son['se'], results_son['sr'], results_son['mrr'],
                              results_son['df_se'], results_son['df_sr'], results_son['df_mrr'], results_son['df_moo'], 'N/A', 'N/A']
                stats_writer.writerow(stats_data)

        if uniform() < PROB_MUTATION and evaluated_points_count < MAX_EVALUATIONS:
            print("\n--- 3. Optional Point (Mutation Operator) ---")
            # Candidate selection: always the best solution found so far

            # Apply mutation operator (multi-point Gaussian perturbation)
            mu = 0.9
            sigma = 0.1
            mutated_X = [best_x_so_far[j] + sigma * random.random() if random.random() <= mu else best_x_so_far[j] for j in range(len(best_x_so_far))]
            mutated_X = [mutated_X[j] - sigma * random.random() if random.random() <= mu else mutated_X[j] for j in range(len(mutated_X))]
            mutated_X = np.clip(mutated_X, lower_bound, upper_bound)

            # Check for duplicate mutated point
            Xsamples_list = np.round(Xsamples, 12).tolist()

            if mutated_copy.tolist() in Xsamples_list:
                print(f"[INFO] Duplicate mutation point detected.")
            else: # Evaluar, actualizar y registrar el nuevo punto mutado
                results_mutation = evaluate_new_point(mutated_X)
                # Imprime el punto, los valores de entrada (SE/SR) y el objetivo
                print(f" 	-> Punto Mutado generado: {np.round(mutated_X, 5)} -> SE: {results_mutation['se']:.5f}, SR: {results_mutation['sr']:.5f} -> Resultado (objetivo): {results_mutation['actual']:.5f}")
                evaluated_points_count += 1
                points_found_this_iteration += 1
                all_results_log.append(results_mutation)
                Xsamples = vstack((Xsamples, mutated_X))
                yactual.append(results_mutation['actual'])

                current_best = min(yactual)
                convergence_data_plot.append(current_best)

                ix = argmin(yactual)
                best_y_so_far = yactual[ix]
                stats_data = [iteration_num, 'mutation', mutated_X[0], mutated_X[1], mutated_X[2], results_mutation['actual'], best_y_so_far,
                              results_mutation['se'], results_mutation['sr'], results_mutation['mrr'],
                              results_mutation['df_se'], results_mutation['df_sr'], results_mutation['df_mrr'], results_mutation['df_moo'], 'N/A', 'N/A']
                stats_writer.writerow(stats_data)

        # Retrain model if new points were found in this iteration
        if points_found_this_iteration > 0:
            print("\n[INFO] Retraining surrogate model with new points...")
            scaler_x = MinMaxScaler().fit(Xsamples)
            rescaledX = scaler_x.transform(Xsamples)
            yactual_arr = asarray(yactual).reshape(-1, 1)
            scaler_y = MinMaxScaler().fit(yactual_arr)
            rescaledY = scaler_y.transform(yactual_arr)
            model.fit(rescaledX, rescaledY.ravel())

        # --- INICIA: NUEVO REPORTE DE FIN DE ITERACIÓN ---
        print(f"\n--- Iteration {iteration_num} Summary ---")

        # Check for new best solution
        ix = argmin(yactual)
        best_y_so_far = yactual[ix]

        if best_y_so_far < best_y_tracker:
            best_x_so_far = Xsamples[ix]
            summary_notification = (f"[NEW BEST] Improved objective: {best_y_so_far:.5f} at X= {np.round(best_x_so_far, 5)}")
            best_y_tracker = best_y_so_far
        else:
            summary_notification = "[STATUS] No improvement in this iteration."

        print(f"New points found: {points_found_this_iteration}")
        print(f"Total evaluations: {evaluated_points_count}")
        print(summary_notification)
        if isinstance(r_squared, float):
            print(f"Surrogate model R-squared: {r_squared:.4f} | RMSE: {rmse:.5f}")

        # Preparamos las variables para la siguiente iteración
        points_found_in_previous_iteration = points_found_this_iteration
        total_points_in_model += points_found_this_iteration
        # --- FIN: NUEVO REPORTE ---

        # Save checkpoint for resuming interrupted runs

        

        state_to_save = {
            'Xsamples': Xsamples,
            'yactual': yactual,
            'all_results_log': all_results_log,
            'model': model,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'iteration_num': iteration_num,
            'evaluated_points_count': evaluated_points_count,
            'best_y_tracker': best_y_tracker,
            'convergence_data_plot': convergence_data_plot,
            'num_initial_points': num_initial_points, # <-- ¡Lo guardamos!
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
        }
        save_checkpoint(state_to_save)

    # (El bucle 'while' termina aquí)
######################## TERMINA BLOQUE CHECKPOINT 5 ########################

    stats_file.close()
    convergence_file.close()

    # Optional: Print detailed evaluation history if checkpoint run is complete
    if checkpoint and (evaluated_points_count >= MAX_EVALUATIONS):
        print("\n" + "="*50)
        print("EVALUATION HISTORY OF COMPLETED RUN")
        print(f"Run completed with {evaluated_points_count} evaluations.")
        print("="*50)

        try:
            # Attempt to import and use IPython display for better visualization
            from IPython.display import display

            # Create table of decision variables (X)
            df_x = pd.DataFrame(Xsamples, columns=['x1 (ap)', 'x2 (fz)', 'x3 (Vc)'])

            # Create table of results (Y)
            df_results = pd.DataFrame(all_results_log)

            # Combine both tables
            df_complete = pd.concat([df_x, df_results], axis=1)

            # Configure pandas display options
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)

            # Print the complete evaluation table
            print(f"Showing all {len(df_complete)} evaluated points (including {num_initial_points} initial LHS points):")
            display(df_complete)

            print("\nFinal optimization summary follows:")

        except Exception as e:
            print(f"[WARNING] Could not display detailed table: {e}")

    # Generate final run report
    final_ix = argmin(yactual)
    final_best_X = Xsamples[final_ix]
    final_best_y = yactual[final_ix]
    final_results = all_results_log[final_ix]

    if final_ix < num_initial_points:
        found_at_message = "Best solution was one of the initial CSV points."
    else:
        evaluation_number = final_ix - num_initial_points + 1
        found_at_message = f"Found during evaluation number: {evaluation_number}"

    print("\n" + "="*50)
    print(f"RUN {run_num + 1} COMPLETED")
    print("="*50)

    print(f"Optimization Mode: {OPTIMIZATION_MODE} (Evaluation: {EVALUATION_MODE})")
    if OPTIMIZATION_MODE == "DF_MOO":
        print(f"   Combined variables: {DF_MOO_VARIABLES}")
    else:
        print(f"   Target variable: {OPTIMIZATION_GOAL}")

    print(f"\nBest objective value found: {final_best_y:.5f}")
    if "DF" in OPTIMIZATION_MODE:
        print(f"   Composite desirability: {-final_best_y:.5f}")

    print(f"   {found_at_message}")
    print(f"Decision variables (ap, fz, Vc): {np.round(final_best_X, 5)}")

    print("\n--- Response Values for Best Solution ---")
    print(f"Surface Finish (SE):         {final_results['se']:.5f}")
    print(f"Tool Stress (SR):            {final_results['sr']:.5f}")
    print(f"Material Removal Rate (MRR): {final_results['mrr']:.5f}")

    print("\n--- Desirability Scores ---")
    print(f"DF_SE:    {final_results['df_se']:.5f}")
    print(f"DF_SR:    {final_results['df_sr']:.5f}")
    print(f"DF_MRR:   {final_results['df_mrr']:.5f}")
    print(f"DF_MOO:   {final_results['df_moo']:.5f}")
    print("-" * 45)

    comp_time_ms = (process_time() - start_time) * 1000
    elapsed_time_ms = (time() - start_act_time) * 1000
    print(f"Total elapsed time: {elapsed_time_ms:.2f} ms")
    print("="*50)

    # Plot convergence trend
    if convergence_data_plot:
        plt.figure(figsize=(10, 6))

        # X-axis: evaluation numbers (1, 2, 3, ...)
        evaluations_axis = range(1, len(convergence_data_plot) + 1)

        plt.plot(evaluations_axis, convergence_data_plot, marker='o', linestyle='-', color='b')

        plt.title('Convergence Graph', fontsize=16)
        plt.xlabel('Number of Evaluations', fontsize=12)
        plt.ylabel('Best Objective Value Found', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Ensure X-axis shows only integer values
        plt.xticks(np.arange(1, len(convergence_data_plot) + 1, 1))

        plt.show()

    best_solution_array.append(final_best_X)
    best_cost_array.append(final_best_y)
    comp_time_array.append(comp_time_ms)
    act_time_array.append(elapsed_time_ms)


# Save summary of all optimization runs
df_runs = pd.DataFrame()
df_runs['Solution'] = pd.Series(best_solution_array)
df_runs['Cost'] = pd.Series(best_cost_array)
df_runs['Comp time (ms)'] = pd.Series(comp_time_array)
df_runs['Elapsed time (ms)'] = pd.Series(act_time_array)
df_runs.to_csv('summary_of_all_runs.csv')

print("\n[INFO] Optimization completed. Summary saved to 'summary_of_all_runs.csv'")
