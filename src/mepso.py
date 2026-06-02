# -*- coding: utf-8 -*-

####################################################################################
# A Particle Swarm Optimization algorithm to find functions optimum.
#
# MEPSO I
# Author: Rafael Batres
# Author: Braulio J. Solano-Rojas
# Institution: Tecnológico de Monterrey
# Date: June 6, 2018. April-May 2022
####################################################################################

from __future__ import annotations

import random
import sys
import copy
import csv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from inspect import signature
from math import isclose
from math import log
from math import exp
from time import process_time
import numpy as np

from datetime import datetime

# For repeatability and reproducibility
#np.random.seed(0)
#random.seed(0)

# Define Chromosome as a subclass of list
class Chromosome(list):
    def __init__(self):
        self.elements = []


# class that represents a particle
class Particle:

    def __init__(self, solution: Chromosome, cost: float):

        # particle is a solution
        self.__solution = solution

        # best solution it has achieved so far by this particle
        self.__best_particle = solution

        # set costs
        self.__new_solution_cost = cost
        self.__best_particle_cost = cost

        # past positions
        self.history = []

    # returns the pbest
    @property
    def best_particle(self) -> Particle:
        return self.__best_particle

    # set pbest
    @best_particle.setter
    def best_particle(self, new_best_particle: Particle):
        self.__best_particle = new_best_particle

    # gets solution
    @property
    def solution(self) -> Chromosome:
        return self.__solution

    # set solution
    @solution.setter
    def solution(self, solution: Chromosome):
        self.__solution = solution

    # gets cost pbest solution
    @property
    def best_particle_cost(self) -> float:
        return self.__best_particle_cost

    # set cost pbest solution
    @best_particle_cost.setter
    def best_particle_cost(self, cost: float):
        self.__best_particle_cost = cost

    # gets cost current solution
    @property
    def current_solution_cost(self) -> Chromosome:
        return self.__new_solution_cost

    # set cost current solution
    @current_solution_cost.setter
    def current_solution_cost(self, cost: float):
        self.__new_solution_cost = cost

    # gets random unique paths - returns a list of lists of paths
    def random_solutions(size: int, search_space: tuple, max_size: int):
        random_solutions = []

        for _ in range(max_size):

            list_temp = Particle.random_solution(size, search_space)

            if list_temp not in random_solutions:
                random_solutions.append(list_temp)

        return random_solutions

    # Generate a random chromosome of continuous values
    def random_solution(size: int, search_space: tuple) -> Chromosome:
        chromosome = Chromosome()
        min, max = search_space
        for i in range(size):
            chromosome.append(np.random.uniform(min[i],max[i]))
        return chromosome


# MicroEPSO algorithm
class MicroEPSO:

    def __init__(self, cost_function, search_space, nvars, iterations: int, max_epochs: int, population_size: int, beta: float=1.0, alfa: float=1.0, mu: float=0.1, sigma: float=0.1, gamma: float=0.1):
        self.cost_function = cost_function  # the cost function
        # number of variables in the cost function
        #self.nvars = len(signature(cost_function).parameters)
        self.nvars = nvars
        self.search_space = search_space  # interval of the cost function
        self.iterations = iterations  # max of iterations
        self.max_epochs = max_epochs
        self.population_size = population_size  # size population
        self.particles = []  # list of particles
        # the probability that all swap operators in swap sequence (gbest - x(t-1))
        self.beta = beta
        # the probability that all swap operators in swap sequence (pbest - x(t-1))
        self.alfa = alfa
        self.__last_epoch = 0
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma

        search_space_min = self.search_space[0]
        search_space_max = self.search_space[1]

        # Selects the best population according to enntropy diversity from
        # Bessaou, M. and Siarry, P. (2001). A genetic algorithm with real-value
        # coding to optimize multimodal continuous functions. Struct Multidisc Optim 23, 63–74
        best_solutions = []
        best_solutions.append(Particle.random_solution(self.nvars, self.search_space))
        h = 0.2 #treshold
        for _ in range(self.population_size-1):
            H = 0.0 # Entropy 
            while H < h:
                chromosome = Particle.random_solution(self.nvars, self.search_space)
                new_solutions = list(best_solutions)
                new_solutions.append(chromosome)
                for j in range(self.nvars):
                    for i in range(len(new_solutions)):
                        sum = 0.0
                        for k in range(1, len(new_solutions)):
                            Pik = 1 - abs(new_solutions[i][j] - new_solutions[k][j])/(search_space_max[j]-search_space_min[j])
                            sum += -Pik * log(Pik)
                        H += sum
                H /= len(new_solutions)
            if H >= h:
                best_solutions.append(chromosome)

        self.__global_best = None
        # initialization of all particles
        for solution in best_solutions:
            # creates a new particle
            particle = Particle(solution=solution,
                                cost=self.cost_function(*solution))
            # add the particle
            self.particles.append(particle)
            # updates gbest if needed
            if self.__global_best is None:
                self.__global_best = copy.deepcopy(particle)
            elif self.__global_best.best_particle_cost > particle.best_particle_cost:
                self.__global_best = copy.deepcopy(particle)

    def init_population(self):
        self.particles = []  # list of particles

        search_space_min = self.search_space[0]
        search_space_max = self.search_space[1]

        # Selects the best population according to enntropy diversity from
        # Bessaou, M. and Siarry, P. (2001). A genetic algorithm with real-value
        # coding to optimize multimodal continuous functions. Struct Multidisc Optim 23, 63–74
        best_solutions = []
        best_solutions.append(Particle.random_solution(self.nvars, self.search_space))
        h = 0.2 #treshold
        for _ in range(self.population_size-1):
            H = 0.0 # Entropy 
            while H < h:
                chromosome = Particle.random_solution(self.nvars, self.search_space)
                new_solutions = list(best_solutions)
                new_solutions.append(chromosome)
                for j in range(self.nvars):
                    for i in range(len(new_solutions)):
                        sum = 0.0
                        for k in range(1, len(new_solutions)):
                            Pik = 1 - abs(new_solutions[i][j] - new_solutions[k][j])/(search_space_max[j]-search_space_min[j])
                            sum += -Pik * log(Pik)
                        H += sum
                H /= len(new_solutions)
            if H >= h:
                best_solutions.append(chromosome)

        # creates the particles
        for solution in best_solutions:
            # creates a new particle
            particle = Particle(solution=solution, cost=self.cost_function(*solution))
            # add the particle
            self.particles.append(particle)

    def evaluate_solutions_diversity(self, solutions: list[Chromosome]) -> float:
        sim_sum = 0
        count = 0
        for solution1 in solutions:
            for solution2 in solutions:
                if not (solution1 == solution2):
                    count += 1
                    # Euclidean distance.  Best distance?
                    sim = euclidean(solution1, solution2)
                    sim_sum += sim
        return sim_sum / count

    # returns gbest (best particle of the population)
    @property
    def global_best(self) -> Particle:
        return self.__global_best

    # set gbest (best particle of the population)
    @global_best.setter
    def global_best(self, new_global_best: Particle):
        self.__global_best = new_global_best

    @property
    def epoch(self) -> int:
        return self.__last_epoch

    @epoch.setter
    def epoch(self, last_epoch: int):
        self.__last_epoch = last_epoch

    # gets solution
    def current_solutions(self) -> list[Chromosome]:
        return [particle.solution for particle in self.particles]

    def mutation_probability(self, initial_probability: float=0.1, current_epoch: int=0, max_epochs: int=1000) -> float:
        alpha = max_epochs/log(initial_probability/10e-03)
        return initial_probability * exp(-current_epoch/alpha)

    def neighborhood_size(self, initial_size: float, current_epoch: int=0, max_epochs: int=1000):
        alpha = max_epochs/log(initial_size/10e-02)
        return initial_size * exp(-current_epoch/alpha)

    # Mutation adding with probability mu a Gaussian perturbation with standard deviation sigma
    def mutate(self, elite_solution: Chromosome, mu: float=0.1, sigma: float=0.1):
        chromosome = [elite_solution[i]+sigma*random.random() if random.random() <= mu else elite_solution[i] for i in range(len(elite_solution))]
        return chromosome

    # Crossover operator
    def crossover(self, dad_chromosome: Chromosome, mom_chromosome: Chromosome, gamma: float=0.1):
        alpha = [random.uniform(-gamma, 1+gamma) for _ in range(len(dad_chromosome))]
        son_chromosome = [alpha[i]*dad_chromosome[i] + (1-alpha[i])*mom_chromosome[i] for i in range(len(dad_chromosome))]
        daugther_chromosome = [alpha[i]*mom_chromosome[i] + (1-alpha[i])*dad_chromosome[i] for i in range(len(dad_chromosome))]
        return son_chromosome, daugther_chromosome

    def run(self):
        # variables for convergence data
        convergence_data = []
        iteration_array = []
        best_cost_array = []
        best_solution_array = []
        epoch_array = []
        epoch_best_cost_array = []
        best_cost_sampling = []
        epoch_best_solution_array =[]

        batch_size = 100  # save data every n iterations
        batch_counter = 0

        HISTORY_SIZE = 100

        epoch = 0
        while epoch < self.max_epochs:
            print("Epoch: ", epoch, "with ", self.population_size, " particles")
            print('Iterations', self.iterations)
            print("Alfa = ", self.alfa, "Beta = ", self.beta)
            convergence_per_epoch = []

            if epoch > 0:
                self.init_population()
                print("Particles: ", len(self.particles))
                mutated_elite = self.mutate(self.__global_best.best_particle, self.mu, self.sigma)
                position = random.randint(0, self.population_size-1)
                self.particles[position] = Particle(mutated_elite, self.__global_best.best_particle_cost)
                print("Inserted elite solution in position", position)

            # for each time step (iteration)
            for t in range(self.iterations):
                convergence_per_iteration = []
                batch_counter = batch_counter + 1

                average_cost = np.mean([particle.best_particle_cost for particle in self.particles])
                cost_std = np.std([particle.best_particle_cost for particle in self.particles])

                # for each particle in the swarm
                for particle in self.particles:
                    previous_cost = particle.current_solution_cost

                    # gets solution of the gbest solution
                    global_best = list(self.__global_best.best_particle)

                    if len(particle.history) == HISTORY_SIZE:
                        particle.history.pop(0)

                    best_neighbor = self.mutate(particle.solution, self.mutation_probability(self.mu, epoch, self.max_epochs), self.sigma)

                    search_space_min = self.search_space[0]
                    search_space_max = self.search_space[1]
                    for i in range(len(best_neighbor)):
                        if best_neighbor[i] < search_space_min[i]:
                            best_neighbor[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                            #best_neighbor[i] = (1 / (1 + exp(best_neighbor[i]))) * search_space_min[i]
                        if best_neighbor[i] > search_space_max[i]:
                            best_neighbor[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                            #best_neighbor[i] = (1 / (1 + exp(-best_neighbor[i]))) * search_space_max[i]

                    best_neighbor_cost = self.cost_function(*best_neighbor)

                    new_solution = particle.solution[:]
                    # gets cost of the current solution
                    new_solution_cost = particle.current_solution_cost

                    if random.random() <= self.beta:
                        new_son_solution, new_daughter_solution = self.crossover(list(new_solution), self.__global_best.best_particle, gamma=self.gamma)

                        for i in range(len(new_son_solution)):
                            if new_son_solution[i] < search_space_min[i]:
                                new_son_solution[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                                #new_son_solution[i] = (1 / (1 + exp(new_son_solution[i]))) * search_space_min[i]
                            if new_daughter_solution[i] < search_space_min[i]:
                                new_daughter_solution[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                                #new_daughter_solution[i] = (1 / (1 + exp(new_son_solution[i]))) * search_space_min[i]
                            if new_son_solution[i] > search_space_max[i]:
                                new_son_solution[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                                #new_son_solution[i] = (1 / (1 + exp(-new_son_solution[i]))) * search_space_max[i]
                            if new_daughter_solution[i] > search_space_max[i]:
                                new_daughter_solution[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                                new_daughter_solution[i] = (1 / (1 + exp(-new_son_solution[i]))) * search_space_max[i]

                        new_son_solution_cost = self.cost_function(*new_son_solution)
                        new_daughter_solution_cost = self.cost_function(*new_daughter_solution)

                        if new_son_solution_cost < new_daughter_solution_cost:
                            new_solution = new_son_solution
                            # gets cost of the current solution
                            new_solution_cost = new_son_solution_cost
                        else:
                            new_solution = new_daughter_solution
                            # gets cost of the current solution
                            new_solution_cost = new_daughter_solution_cost

                    elif random.random() <= self.alfa:
                        largest_dist = 0
                        for neighbor_particle in self.particles:
                            sol = neighbor_particle.best_particle
                            dist = euclidean(global_best, sol)

                            if dist > largest_dist:
                                largest_dist = dist
                                dissimilar_particle = neighbor_particle

                        new_son_solution, new_daughter_solution = self.crossover(list(new_solution), dissimilar_particle.best_particle, gamma=self.gamma)

                        for i in range(len(new_son_solution)):
                            if new_son_solution[i] < search_space_min[i]:
                                new_son_solution[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                                #new_son_solution[i] = (1 / (1 + exp(new_son_solution[i]))) * search_space_min[i]
                            if new_daughter_solution[i] < search_space_min[i]:
                                new_daughter_solution[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                                #new_daughter_solution[i] = (1 / (1 + exp(new_daughter_solution[i]))) * search_space_min[i]
                            if new_son_solution[i] > search_space_max[i]:
                                new_son_solution[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                                #new_son_solution[i] = (1 / (1 + exp(-new_son_solution[i]))) * search_space_max[i]
                            if new_daughter_solution[i] > search_space_max[i]:
                                new_daughter_solution[i] = random.random() * (search_space_max[i]-search_space_min[i]) + search_space_min[i]
                                new_daughter_solution[i] = (1 / (1 + exp(-new_daughter_solution[i]))) * search_space_max[i]



                        new_son_solution_cost = self.cost_function(*new_son_solution)
                        new_daughter_solution_cost = self.cost_function(*new_daughter_solution)

                        if new_son_solution_cost < new_daughter_solution_cost:
                            new_solution = new_son_solution
                            # gets cost of the current solution
                            new_solution_cost = new_son_solution_cost
                        else:
                            new_solution = new_daughter_solution
                            # gets cost of the current solution
                            new_solution_cost = new_daughter_solution_cost

                    if new_solution_cost < best_neighbor_cost:
                        best_neighbor = new_solution[:]
                        best_neighbor_cost = new_solution_cost

                    if best_neighbor_cost < previous_cost and best_neighbor not in particle.history:
                        # updates the current solution
                        particle.solution = best_neighbor
                        # updates the cost of the current solution
                        particle.current_solution_cost = best_neighbor_cost
                        particle.history.append(best_neighbor)

                    # checks if new solution is pbest solution
                    pbCost = particle.best_particle_cost

                    if best_neighbor_cost < pbCost:
                        particle.best_particle = best_neighbor
                        particle.best_particle_cost = best_neighbor_cost

                    gbestCost = self.__global_best.best_particle_cost

                    # check if new solution is gbest solution
                    if particle.current_solution_cost < gbestCost:
                        self.__global_best = copy.deepcopy(particle)

                #if batch_counter > batch_size:
                #print("Sum of acceptance probabilities:", sumAcceptanceProbabilities)
                print(t, "Gbest cost = ", self.__global_best.best_particle_cost)
                convergence_per_iteration.append(t)
                convergence_per_iteration.append(self.__global_best.best_particle_cost)
                #convergence_per_iteration.append(average_cost)
                #convergence_per_iteration.append(cost_std)
                convergence_data.append(convergence_per_iteration)
                iteration_array.append(t)
                best_cost_array.append(self.__global_best.best_particle_cost)
                best_solution_array.append(self.__global_best.best_particle)
                #batch_counter = 0

                #if self.max_epochs > 1:
                convergence_per_epoch.append(epoch)
                convergence_per_epoch.append(self.__global_best.best_particle_cost)
                convergence_data.append(convergence_per_epoch)
                epoch_array.append(epoch)
                epoch_best_cost_array.append(self.__global_best.best_particle_cost)
                epoch_best_solution_array.append(self.__global_best.best_particle)

            epoch = epoch + 1
            self.epoch = epoch
            best_cost_sampling.append(self.__global_best.best_particle_cost)
            if epoch > 5:
                std = np.std(best_cost_sampling[-10:])
                print("standard deviation: ", std)
            else:
                std = 1000

            if isclose(std, 0):
                break

        #print("What's going on?")
        print("Cost of global best: ", self.__global_best.best_particle_cost)
        #print("global best: ", self.__global_best.best_particle)
        print("")
        #now = datetime.now()
        #date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        
        #convergence_file_name = "epoch_convergence" + date_time + ".csv"
        #df_epoch = pd.DataFrame()
        #df_epoch['Epoch'] = pd.Series(epoch_array)
        
        #df_epoch['Best cost'] = pd.Series(epoch_best_cost_array)
        #df_epoch['Best solution'] = pd.Series(epoch_best_solution_array)
        #df_epoch.to_csv(convergence_file_name, index=False)
        

