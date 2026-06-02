# -*- coding: utf-8 -*-
"""
MEVO: Meta-Evolutionary Optimizer with Surrogate-Based Optimization

This module implements an adaptive surrogate-based optimization algorithm that combines:
1. Gaussian Process (GP) or ANN surrogate models for function approximation
2. MicroEPSO (Micro Evolutionary Particle Swarm Optimizer) for acquisition function optimization
3. Evolutionary operators (crossover and mutation) for generating new candidate solutions
4. Active learning strategies with adaptive search space reduction

The algorithm iteratively:
- Samples new points by optimizing an acquisition function
- Updates the surrogate model with actual function evaluations
- Reduces the search space based on elite solutions
- Maintains an archive of evaluated solutions to avoid redundant evaluations

Author: Rafael Batres
Institution: Tecnologico de Monterrey
Date: June 4, 2025

References:
    - Batres, et al. (2023). MEVO: A Metamodel-Based Evolutionary Optimizer for Building Energy Optimization
    - https://www.mdpi.com/1996-1073/16/20/7026
"""

####################################################################################
# IMPORTS
####################################################################################

from __future__ import annotations

import pandas as pd
from time import process_time, time
from datetime import datetime
import os
import random
import copy
import csv
from math import exp, sqrt
import numpy as np

# Numerical computing
import math
from numpy import vstack, argmin, asarray
from numpy.random import normal, uniform
from scipy.stats import norm

# Machine learning and surrogate modeling
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as ConstantKer
from warnings import catch_warnings, simplefilter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans

# Visualization and utilities
import matplotlib.pyplot as plt
from initial_sampling_plan import initial_sampling_plan

# MEPSO import
from mepso import *

import subprocess
import json
import ast
from scipy.optimize import differential_evolution





####################################################################################
# PROBLEM CONFIGURATION
####################################################################################

best_solution_array = []
ndim = 2  # Number of dimensions
nvars = ndim
for i in range(nvars):
    best_solution_array.append([])

n_samples = 30  # Initial number of samples for surrogate training
no_of_replications = 1  # Number of function evaluations per sample (replications)

obj_func_search_space = [-5, 10]  # Search space [lower_bound, upper_bound]
# Define lower and upper bounds for each dimension
lower_bound = [obj_func_search_space[0]] * ndim
upper_bound = [obj_func_search_space[1]] * ndim

original_lower_bound = lower_bound[:]
original_upper_bound = upper_bound[:]

# runs the algorithm number_of_runs times
number_of_runs = 1


####################################################################################
# OBJECTIVE FUNCTION DEFINITION
####################################################################################

def obj_func(x1, x2):
    """
    Rosenbrock function (2D benchmark function for optimization).
    
    A classic benchmark function used to test optimization algorithms.
    Global optimum at (1, 1) with value 0.
    
    Args:
        x1: First variable
        x2: Second variable
    
    Returns:
        float: Function value = 100(x2 - x1^2)^2 + (x1 - 1)^2
    """
    sum = 0.0
    sum += 100 * (x2 - x1**2)**2 + (x1 - 1)**2
    return sum



####################################################################################
# UTILITY FUNCTIONS - Statistical Measures
####################################################################################

def current_milli_time():
    """Get current time in milliseconds."""
    return round(time() * 1000)

def variance(data, ddof=0):
    """
    Calculate variance of data.
    
    Args:
        data: List of numerical values
        ddof: Degrees of freedom (default: 0 for population variance)
    
    Returns:
        float: Variance of the data
    """
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)

def stdev(data):
    """
    Calculate standard deviation of data.
    
    Args:
        data: List of numerical values
    
    Returns:
        float: Standard deviation of the data
    """
    var = variance(data)
    std_dev = sqrt(var)
    return std_dev

def mean(data):
    """
    Calculate arithmetic mean of data.
    
    Args:
        data: List of numerical values
    
    Returns:
        float: Mean of the data
    """
    n = len(data)
    mean = sum(data) / n
    return mean

# Function to get the first n columns as a NumPy array
def get_first_n_columns_as_numpy(df, n):
    return df.iloc[:, :n].to_numpy()


####################################################################################
# VISUALIZATION FUNCTION
####################################################################################

def plot_results(data1, data2, name):
    
    # Plot target curve and fitted curve
    discretization1 = data1[:, 0]
    stress1 = data1[:, 1]
    
    discretization2 = data2[:, 0]
    stress2 = data2[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(discretization1, stress1, label='Curve 1')
    plt.plot(discretization2, stress2, label='Curve 2')
    plt.xlabel('Time [s]')
    plt.ylabel('Stress [MPa]')
    plt.xscale('log')
    plt.xlim(0.3,10000)
    plt.legend()
    plt.grid()
    plt.savefig(name)
    plt.close('all')

####################################################################################
# SURROGATE MODEL
####################################################################################

# Neural Network Model
model = MLPRegressor(hidden_layer_sizes=8, activation='logistic', solver='lbfgs',
                         max_iter=1500, learning_rate='adaptive', learning_rate_init=0.001,
                         random_state=42)

# Gaussian Process Model
#kernel = ConstantKer(1.0, (1e-3, 1e3)) * RBF([5,5,5], (1e-2, 1e2))
#model = GaussianProcessRegressor(kernel=Matern(nu=1.5), alpha = 0.0001, n_restarts_optimizer=100)



# surrogate or approximation for the objective function 
def surrogate(model, x):
    global lower_bound, upper_bound
    global scl
    global scaler_x, scaler_y
    Sol = [x]
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
                            
        std = 0
        
        rescaledX = scaler_x.transform(Sol)
        surr_prediction = model.predict(rescaledX)
        surr_prediction_arr = asarray(surr_prediction)
      
        surr_prediction = surr_prediction_arr.T
        surr_prediction = scaler_y.inverse_transform(surr_prediction.reshape(1, -1))
        
        reg_param_small=5e-2
        reg_param_large=5e-2
        
        # Ensure lower_bound and upper_bound are numpy arrays
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)

        regularization_small =  reg_param_small * np.sum(np.maximum(0, np.exp(-(x - lower_bound))))
        regularization_large =  reg_param_large * np.sum(np.maximum(0, np.exp(-(upper_bound-x))))
        
        total_regularization = regularization_small + regularization_large
        
        surr_prediction += total_regularization
        
        return surr_prediction, std

####################################################################################
# ACQUISITION FUNCTION
####################################################################################

def acquisition_ei(x1, x2, x3):
    X = []
    global Xsamples
    global model
    global yactual
    global ysurrogate
    bestmu = min(ysurrogate)
    # calculate the surrogate score
    yhat, _ = surrogate(model, x1, x2, x3)
    ysurrogate_plus_yhat = ysurrogate[:]
    ysurrogate_plus_yhat.append(yhat)
    std_mu = stdev(ysurrogate_plus_yhat)
    mse = 0
    for i in range(len(Xsamples)):
      mse += ((yhat - ysurrogate[i])**2)

    if bestmu - yhat > 0:
      quant = float((bestmu - yhat) / (1E-9 + std_mu))

      total = 1/(float(bestmu - yhat)*norm.cdf(quant) + float(mse)*norm.pdf(quant))
    else:
      total = 1E999
    return total


# Acquisition function wrapper for optimizer
def acquisition(*decVars):
    global model
    global ysurrogate
    new_x = []
    for dec_var in decVars:
        new_x.append(dec_var)
    yhat, _ = surrogate(model, new_x)
    return yhat


####################################################################################
# OPTIMIZATION OF THE ACQUISITION FUNCTION
####################################################################################

# Optimize the acquisition function
def opt_acquisition_3d(X, model, lower_bound, upper_bound):
    #global lower_bound, upper_bound
    # Uses MEPSO to optimize the acquisition function
    pso = MicroEPSO(acquisition, (lower_bound, upper_bound),ndim,
                iterations=15, # iterations (inside loop) #15
                max_epochs=2, # max epochs (outside loop) #10  # org= 2
                population_size=20, # population size # 15  # org 20
                beta=0.9, # beta is the probability for a movement based on the global best
                alfa=0.6, # alfa is the probabiliy for a movement based on local best 0.6
                mu=0.5, # 0.1 Mutation adding with probability mu a Gaussian perturbation 0.5
                sigma=0.7, # with standard deviation sigma
                gamma=0.7) # 0.3 percentage of value taken from one parent in crossover
                            # gamma=0 means siblings are equal to parents. Default: 0.7
    pso.run()  # runs the MicroEPSO algorithm
    Xopt = pso.global_best.best_particle
    print("Global best: ", Xopt)
    return Xopt




# For recording data of each run
best_cost_array = []
comp_time_array = []
act_time_array  = []
simulations_array = []



#Initial sampling plan generation
response = int(input('1 to generate new sampling plan, 2 to omit it: '))
if response == 1: 
    response_2 = int(input('Are you sure to generate the new sampling plan? 1 to confirm, 2 to cancel: '))
    if response_2 == 1:
        initial_sampling_plan(ndim, n_samples, upper_bound, lower_bound, obj_func)
    elif response == 2:
        print('Using same sampling plan')
elif response == 2: 
    print('Using same sampling plan')



now = datetime.now()
path = f'./run_{now.strftime("%m_%d_%Y_%H_%M_%S")}'
if not os.path.exists(path):
    os.mkdir(path)
                          
                  
for _ in range(number_of_runs):



    bestX  = []
    bestY = 1E-300                                      

    Xsamples = []
    bestX  = []
    yactual = []
    ysurrogate = []

    start_act_time = current_milli_time()
    start_time = process_time()

    # When reading initial points
    Xsamples = []
    df = pd.read_csv('initial_points.csv')

    Xsamples_df = get_first_n_columns_as_numpy(df, ndim)

    data_x1 = []
    data_x2 = []
    data_x3 = []
    for i in range(len(df)):
        data_x = []
        data_x1.append(df.iloc[i, 0])
        data_y = df.iloc[i, ndim]
        yactual.append(data_y)

    Xsamples = Xsamples_df                     
    print("Xsamples: ", Xsamples)
    print("y:", yactual)

    # ********
    # Number of clusters
    k = 3

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)

    # ********

    # Update best solution data
    ix = argmin(yactual)
    bestX = Xsamples[ix][:]
    bestX = bestX.tolist()
    bestY = yactual[ix]
    print("Best X", bestX)
    print('Best Result (y): y=%.3f' % bestY)

    # Memory for retrieving already simulated solutions
    Xmemory = copy.copy(Xsamples)
    Ymemory = copy.copy(yactual)                        



    # transofrm data
    scaler_x = MinMaxScaler(feature_range = (0, 1)).fit(Xsamples)
    rescaledX = scaler_x.transform(Xsamples)
                                   
    yactual_arr = asarray(yactual)
    scaler_y = MinMaxScaler(feature_range = (0,1)).fit(yactual_arr.reshape(len(yactual_arr),1))
    rescaledY = scaler_y.transform(yactual_arr.reshape(len(yactual_arr),1))




    # Generate the model
    model.fit(rescaledX, rescaledY.ravel())

    #"""

    # Display the model parameters 
    """    
    coeffs = model.coefs_
    biases = model.intercepts_
    total_length = sum(len(row) for row in coeffs)
    no_layers = len(coeffs)
    no_neurons_per_layer = len(coeffs[0])
    no_weights = 0
    no_biases = 0
    for layer in range(no_layers):
        no_weights += sum(len(row) for row in coeffs[layer])
                                                            

    for layer in range(len(biases)):
        no_biases += len(biases[layer])

    param_num = no_weights + no_biases
    print("ANN weights: ", coeffs)
    print("Number of weights: ", no_weights)
    print("ANN biases: ", biases)
    print("Number of biases: ", no_biases)
    print("Total number of ANN paramters: ", param_num)
    # mean squared error
    y_pred = model.predict(rescaledX)
    mse = mean_squared_error(rescaledY, y_pred)
    rmse = sqrt(mse)
    aic = len(Xsamples) * math.log(mse) + 2*param_num
    print("MSE: ", round(mse,6), math.log(mse), math.log(math.e))
    print("RMSE: ", round(rmse,4))
    r2_squared = r2_score(rescaledY, y_pred)
                                           
    print('R-squared score:', round(r2_squared,4))
    print("Number of data points: ", len(Xsamples))
    print("Akaike Information Criterion:", round(aic, 4))
    """

    """
    # Ensemble model
    level0_f1 = list()
    level0_f1.append(('svr_f1', SVR(kernel=1.41**2 * RBF(length_scale=0.1))))
    level0_f1.append(('gb_f1', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)))
    level1_f1 = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0_f1, final_estimator=level1_f1, cv = KFold(n_splits=2, shuffle=False))
    model.fit(rescaledX, rescaledY.ravel()) 
    """

    # Calculate the response using the surrogate model
    for x in Xsamples:
        est, _ = surrogate(model, x)
        ysurrogate.append(est)   
           
    best_x1 = []
    best_x2 = []
    best_x3 = []

    # Save the results in a file
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    stats_file_name = os.path.join(path, "statistics" + date_time + ".csv")
    stats_file = open(stats_file_name, 'w', encoding='UTF8', newline='')
    stats_writer = csv.writer(stats_file)
    header = []
    for i in range(nvars): 
        x_val = f'X{i+1}'
        header.append(x_val)
    header.append('Simulation Value')
    header.append('Surrogate Value')
    header.append('Best Result')
    header.append('RMSE')
    header.append('R^2')
    stats_writer.writerow(header)

    x1_values = []
    x2_values = []
    x3_values = []

    data = []

    convergence_file_name = os.path.join(path, "convergence" + date_time + ".csv")
    convergence_file = open(convergence_file_name, 'w', encoding='UTF8', newline='')
    convergence_writer = csv.writer(convergence_file)
    convergence_header = []

    for i in range(nvars):
        header_val = f'X{i+1}'
        convergence_header.append(header_val)
    convergence_header.append('Best Result')
    
    convergence_writer.writerow(convergence_header) 
    convergence_curve = []
    convergence_curve = copy.copy(bestX)
    convergence_curve.append(bestY)
    convergence_writer.writerow(convergence_curve)
    convergence_curve = []

    added_samples = []
    # The main loop starts here
    simulations = 0
    max_simulations = 280
    duplicates_allowed = True # True

    # Size of the data set (archive) for the surrogate model generation
    data_set_size = 100 # 100
                     
    iter = 0
    while simulations < max_simulations:
        iter += 1
        print("*************")
        print("ITERATION: ", i)
        print("*************")        
        # Optimize the acquisition function to find the new sampling point x
        x = opt_acquisition_3d(Xsamples, model, lower_bound, upper_bound)
        x_copy = asarray(x)
        
        # Order the solutions
        best_n_sols = []
        # changed reverse=True to order values from larger to lower
        ordered_y = sorted(yactual, key = lambda y:float(y), reverse=False)
        best_n = len(Xsamples)
            
        for j in range(best_n):
            index = np.where(yactual==ordered_y[j])[0][0]
            best_n_sols.append(Xsamples[index])

        Xsamples = best_n_sols[:]
        Xsamples = asarray(Xsamples)
        yactual = ordered_y[:]      

        # Keep the length of Xsamples constant by removing the worst data
        while len(Xsamples) > data_set_size:
            Xsamples = np.delete(Xsamples, -1, 0)
            yactual = np.delete(yactual, -1, 0)     

        # Reduce the size of the search space
        d = np.array(upper_bound) - np.array(lower_bound)
        alpha = 0.8 # 0.8
        s_star = []
        max_values = np.array(best_n_sols).max(axis = 0)
        max_values = np.array(max_values) + alpha*d/2.0
        min_values = np.array(best_n_sols).min(axis = 0)
        min_values = np.array(min_values) - alpha*d/2.0
        print("min and max values: ", min_values, max_values)
        for var_index in range(nvars):
            upper_bound[var_index] = min(max_values[var_index], upper_bound[var_index])
            upper_limit_ext = lower_bound[var_index]+1
            upper_bound[var_index] = max(upper_bound[var_index], upper_limit_ext)
            lower_bound[var_index] = max(min_values[var_index], lower_bound[var_index])
            if upper_bound[var_index] > original_upper_bound[var_index]:
                upper_bound[var_index] = original_upper_bound[var_index]
            if lower_bound[var_index] < original_lower_bound[var_index]:
                lower_bound[var_index] = original_lower_bound[var_index]
        lower_bound = [elem for elem in lower_bound]
        upper_bound = [elem for elem in upper_bound]
        print("New boundaries: ",  lower_bound, upper_bound,)            
        
        # sample points already simulated are not simulated again
        x_copy [0] = round(x[0], 12)
        Xlist = Xsamples.tolist()        
        
        tol = 1e-6
        diff = (Xmemory - x_copy)
        np_where_result = np.where((abs(diff) < tol).all(-1))
        exists = np_where_result[0]
        already_simulated_solution = False
        if exists.size > 0:
            x_index = exists[0]
            already_simulated_solution = True                   
            actual = Ymemory[x_index]
            print("x: ", x, "y: ", actual, "(an already simulated point)")
        else:
            # if x has not been simulated then run the simulation
            sumactual=0
            for i in range (no_of_replications):            
                actual = obj_func(*x_copy)
                sumactual += actual
                simulations += 1
            actual= sumactual/no_of_replications

            if simulations == max_simulations:
                break

        x_copy_as_list = x_copy.tolist()                              
        print("Newly added point:", x)

        # Update the best solution found so far
        if actual < bestY:
            print("Previous best: ", bestY)
            bestY = actual
            bestX = x_copy_as_list
            print("New best: ", bestY)                          

        x1_values.append(x[0])
        est, MSE = surrogate(model, x_copy)
        Xsamples_org = Xsamples.copy()
        yactual_org = yactual[:]

        Xsamples = vstack((Xsamples, x_copy_as_list))
        yactual = np.append(yactual, actual)

        if already_simulated_solution == False or duplicates_allowed:
            print("Memory has been updated ...")
            Xmemory = vstack((Xmemory, x_copy_as_list))
            Ymemory = np.append(Ymemory, actual)
        added_samples.append(x_copy_as_list)

           

        # Scale again as a new point has been added
        scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(Xsamples)
        rescaledX = scaler_x.transform(Xsamples)

        yactual_arr = asarray(yactual)
                             
        scaler_y = MinMaxScaler(feature_range = (0,1)).fit(yactual_arr.reshape(len(yactual_arr),1))
        rescaledY = scaler_y.transform(yactual_arr.reshape(len(yactual_arr),1))

        model.fit(rescaledX, rescaledY.ravel())

        print("Best X", bestX)
        print("Best Result (Y):", bestY)
        best_result_so_far = bestY
        
        for i in range(nvars):
            convergence_curve.append(bestX[i])               

        convergence_curve.append(bestY)
        convergence_writer.writerow(convergence_curve)
        convergence_curve = []
        
        new_est, MSE = surrogate(model, x_copy)
        new_est = new_est.item()
        new_est = float(new_est)
        print("x: ", x, "y_est= ", new_est, "y_actual= ", actual)

        ysurrogate.append(new_est)
        y_pred = model.predict(rescaledX)
        mse = mean_squared_error(rescaledY, y_pred)
        rmse = sqrt(mse)
        print("RMSE =", rmse)
        # r-squared
        r_squared = r2_score(rescaledY, y_pred)

        print("R-squared score: ", round(r_squared,4))

        if r_squared < 0.1:
            Xsamples = Xsamples_org
            yactual = yactual_org
            scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(Xsamples)
            rescaledX = scaler_x.transform(Xsamples)

            yactual_arr = asarray(yactual)
            scaler_y = MinMaxScaler(feature_range = (0,1)).fit(yactual_arr.reshape(len(yactual_arr),1))
            rescaledY = scaler_y.transform(yactual_arr.reshape(len(yactual_arr),1))
        else:
            data = copy.copy(x_copy_as_list)
            data.append(actual)
            data.append(new_est)
            best_result_so_far = bestY
            data.append(bestY)
            data.append(rmse)
            data.append(r_squared)
            stats_writer.writerow(data)

        # Rank the solutions
        best_n_sols = []
        # changed reverse=True to order values from larger to lower
        ordered_y = sorted(yactual, key = lambda y:float(y), reverse=False)
        print("ordered y", ordered_y)
        best_n = round(0.1*len(Xsamples))
            
        for j in range(best_n):
            index = np.where(yactual==ordered_y[j])[0][0]
            #index = yactual.index(ordered_y[j])
            best_n_sols.append(Xsamples[index])
        #best_n_sols.append(bestX)




        # Do a crossover and update the surrogate model
        # Sep 18 2022: 0.1
        if uniform() < 0.790199:
            k = 0
            while True:
                chosen_dad = random.randint(0,best_n - 1)
                dad = best_n_sols[chosen_dad]
                chosen_mom = random.randint(0,best_n - 1)
                mom = best_n_sols[chosen_mom]
                if list(dad) != list(mom) or k == best_n:
                    break
                k += 1

            gamma = 0.7
            alpha = [random.uniform(-gamma, 1+gamma) for _ in range(len(dad))]
            son = [alpha[i]*dad[i] + (1-alpha[i])*mom[i] for i in range(len(dad))]

            for i in range(len(son)):

                if son[i] < lower_bound[i]:

                    son[i] = (1 / (1 + exp(son[i]))) * lower_bound[i]
                if son[i] > upper_bound[i]:

                    son[i] = (1 / (1 + exp(-son[i]))) * upper_bound[i]
            print("\n Newly added point (from crossover):", son)
            son_copy = asarray(son)

            tol = 1e-8
            diff = (Xmemory - son_copy)
            np_where_result = np.where((abs(diff) < tol).all(-1))
            exists = np_where_result[0]
            already_simulated_solution = False
            if exists.size > 0:
                x_index = exists[0]
                son_response = Ymemory[x_index]
                already_simulated_solution = True
                print("x: ", son, "y: ", son_response, "(an already simulated point)")
            # if x has not been simulated then run the simulation
           
            else:
                sumson=0
                for i in range (no_of_replications): 
                    son_response = obj_func(*son_copy)
                    sumson += son_response
                    simulations += 1
                son_response = sumson/no_of_replications

                if simulations == max_simulations:
                    break
                                        
                                
            # Update the best solution found so far
            son_copy_as_list = son_copy.tolist()            
            if son_response < bestY:
                bestY = son_response
                bestX = son_copy_as_list                                                

            # Update sampling plan
            if already_simulated_solution == False or duplicates_allowed:

                Xsamples = vstack((Xsamples, son_copy_as_list))
                yactual = np.append(yactual, son_response)

                added_samples.append(son_copy_as_list)
                crossover_est, _ = surrogate(model, son_copy_as_list)
                crossover_est = crossover_est.item()
                ysurrogate.append(float(crossover_est))
                data = copy.copy(son_copy_as_list)
                data.append(son_response)
                data.append(float(crossover_est))
                if son_response < best_result_so_far:
                    best_result_so_far = son_response
                                 
                                      
                # Keep the length of Xsamples constant by removing oldest data
                #if len(Xsamples) > data_set_size:
                    #Xsamples = np.delete(Xsamples, 0, 0)
                    #yactual = np.delete(yactual, 0, 0)
                    ##Xsamples = np.delete(Xsamples, -1, 0)
                    ##yactual = np.delete(yactual, -1, 0)                       

                data.append(bestY)
                data.append(rmse)
                data.append(r_squared)
                stats_writer.writerow(data)

        #r = uniform()
 

        # Do a mutation and update the surrogate model

        if uniform() < 0.779537:
            mu = 0.9 
            sigma = 0.1
            print("Best X is going to be mutated: ", bestX)
            # Implements Gaussian mutation

            mutated_X = [bestX[i]+sigma*random.random() if random.random() <= mu else bestX[i] for i in range(len(bestX))]          
            mutated_X = [mutated_X[i]-sigma*random.random() if random.random() <= mu else mutated_X[i] for i in range(len(mutated_X))]

            #mutated_X = polynomial_mutation_vector(bestX, 20, lower_bound, upper_bound)
            print("Mutated_x: ", mutated_X)
            for i in range(len(mutated_X)):
                if mutated_X[i] < lower_bound[i]:
                    mutated_X[i] = (1 / (1 + exp(mutated_X[i]))) * lower_bound[i]
                if mutated_X[i] > upper_bound[i]:
                    mutated_X[i] = (1 / (1 + exp(-mutated_X[i]))) * upper_bound[i]

            if mutated_X[0] == 50:
                exit()
            print("Newly added point (from mutation):", mutated_X)

            # sample points already simulated are not simulated again
            mutated_X = asarray(mutated_X)
            diff = (Xmemory - mutated_X)
            np_where_result = np.where((abs(diff) < tol).all(-1))
            exists = np_where_result[0]
            already_simulated_solution = False
            if exists.size > 0:
                x_index = exists[0]
                already_simulated_solution = True           
                mutated_sol = Ymemory[x_index]
                print("x: ", mutated_X, "y: ", mutated_sol, "(an already simulated point)")
                
            # if x has not been simulated then run the simulation
            else:
                summut=0
                for i in range (no_of_replications): 
                    mutated_X_response = obj_func(*mutated_X)
                    summut += mutated_X_response
                    simulations += 1
                mutated_X_response = summut/no_of_replications

                new_point = True                         

                if simulations == max_simulations:
                    break
                
            # Update sampling plan
            mutated_X_as_list = mutated_X.tolist()
            if mutated_X_response < bestY:
                bestY = mutated_X_response
                bestX = mutated_X_as_list                                                                      

            if already_simulated_solution == False or duplicates_allowed:
                Xsamples = vstack((Xsamples, mutated_X_as_list))
                yactual = np.append(yactual, mutated_X_response)
                added_samples.append(mutated_X_as_list)
                mutated_est, _ = surrogate(model, mutated_X_as_list)
                mutated_est = mutated_est.item()                                            
                ysurrogate.append(float(mutated_est))
                                                        
                data = copy.copy(mutated_X_as_list)
                data.append(mutated_X_response)
                data.append(float(mutated_est))
                if mutated_X_response < best_result_so_far:
                    best_result_so_far = mutated_X_response

                     

                data.append(bestY)
                data.append(rmse)
                data.append(r_squared)
                stats_writer.writerow(data)

            if simulations == max_simulations:
                break

    stats_file.close()
    convergence_file.close()

    # Report the final results
    print("size of y:", len(yactual))
    print("Best Result (X): ", bestX)
    print("Best Result (y): ", bestY)
                                     
    ms = (process_time() - start_time) * 1000.0
    elapsed_time = (current_milli_time() - start_act_time)
    print("Elapsed time: ", elapsed_time)
    for i in range(nvars): 
        best_solution_array[i].append(bestX[i])
    best_cost_array.append(bestY)
    comp_time_array.append(ms)
    act_time_array.append(elapsed_time)
    simulations_array.append(simulations)
    scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(Xsamples)
    rescaledX = scaler_x.transform(Xsamples)

    y_pred = model.predict(rescaledX)
    yactual_arr = asarray(yactual)
    scaler_y = MinMaxScaler(feature_range = (0,1)).fit(yactual_arr.reshape(len(yactual_arr),1))
    rescaledY = scaler_y.transform(yactual_arr.reshape(len(yactual_arr),1))   
    for k in range(len(y_pred)):
            ydash = scaler_y.inverse_transform(y_pred[k].reshape(1, -1))
            print(yactual[k], ",", ydash[0][0]) 
            

    
    identified_params = bestX
    

# save the results of all the runs
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
df = pd.DataFrame()


for i in range(len(best_solution_array)):
    df[f'Solution {i + 1}'] = pd.Series(best_solution_array[i])
df['Cost'] = pd.Series(best_cost_array)
df['Comp time'] = pd.Series(comp_time_array)
df['Elapsed time'] = pd.Series(act_time_array)
df['Simulations'] = pd.Series(simulations_array)
runs_file_name = os.path.join(path, "runs_" + date_time + ".csv")
df.to_csv(runs_file_name)