import streamlit as st
import csv
import random
import pandas as pd

def fitness_function(solution):
    return sum(x**2 for x in solution)

def initialize_population(pop_size, dimensions, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, dimensions))

def categorize_market(population, market_type):
    if market_type == "balanced":
        groups = [
            int(0.25 * len(population)),
            int(0.50 * len(population)),
            int(0.25 * len(population))
        ]
    elif market_type == "fluctuating":
        groups = [
            int(0.20 * len(population)),
            int(0.60 * len(population)),
            int(0.20 * len(population))
        ]
    return groups

def risk_based_update(population, risk_factor, group_indices):
    for i in group_indices:
        noise = np.random.uniform(-risk_factor, risk_factor, size=population.shape[1])
        population[i] += noise

def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    return np.concatenate((parent1[:point], parent2[point:]))

def mutate(solution, mutation_rate, lower_bound, upper_bound):
    for i in range(len(solution)):
        if np.random.rand() < mutation_rate:
            solution[i] += np.random.uniform(-mutation_rate, mutation_rate)
            solution[i] = np.clip(solution[i], lower_bound, upper_bound)
    return solution

def optimize(pop_size, dimensions, max_generations, lower_bound, upper_bound, risk_factors, mutation_rate):
    population = initialize_population(pop_size, dimensions, lower_bound, upper_bound)
    best_solution = None
    best_fitness = float('inf')

    for generation in range(max_generations):
        fitness_scores = np.array([fitness_function(ind) for ind in population])
        sorted_indices = np.argsort(fitness_scores)
        population = population[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]

        if fitness_scores[0] < best_fitness:
            best_fitness = fitness_scores[0]
            best_solution = population[0]

        market_type = "balanced" if generation % 2 == 0 else "fluctuating"
        groups = categorize_market(population, market_type)
        
        risk_based_update(population[groups[1]:groups[1] + groups[2]], risk_factors[0], range(groups[1], groups[1] + groups[2]))
        risk_based_update(population[groups[2]:], risk_factors[1], range(groups[2], len(population)))

        next_population = []
        for i in range(pop_size // 2):
            parent1, parent2 = population[np.random.randint(0, pop_size, 2)]
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring, mutation_rate, lower_bound, upper_bound)
            next_population.append(offspring)

        population = np.array(next_population)

        if generation % 10 == 0 or generation == max_generations - 1:
            print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

# Parameters
pop_size = 50
dimensions = 10
max_generations = 100
lower_bound = -1
upper_bound = 1
risk_factors = [0.1, 0.2]
mutation_rate = 0.05

# Run Optimization
best_solution, best_fitness = optimize(pop_size, dimensions, max_generations, lower_bound, upper_bound, risk_factors, mutation_rate)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
