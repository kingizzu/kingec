import numpy as np
import streamlit as st

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
    history = []

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

        history.append(best_fitness)

        if generation % 10 == 0 or generation == max_generations - 1:
            st.write(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness, history

# Streamlit Interface
st.title("Hybrid Optimization Algorithm (EMA + GA)")

# Parameters
pop_size = st.number_input("Population Size", min_value=10, value=50)
dimensions = st.number_input("Dimensions", min_value=2, value=10)
max_generations = st.number_input("Maximum Generations", min_value=10, value=100)
lower_bound = st.number_input("Lower Bound", value=-1.0)
upper_bound = st.number_input("Upper Bound", value=1.0)
risk_factors = [
    st.number_input("Risk Factor for Group 2", value=0.1),
    st.number_input("Risk Factor for Group 3", value=0.2)
]
mutation_rate = st.number_input("Mutation Rate", min_value=0.01, value=0.05)

if st.button("Run Optimization"):
    best_solution, best_fitness, history = optimize(pop_size, dimensions, max_generations, lower_bound, upper_bound, risk_factors, mutation_rate)
    st.success(f"Optimization Completed! Best Fitness: {best_fitness}")
    st.write("Best Solution:", best_solution)

    # Plot convergence
    st.line_chart(history)
