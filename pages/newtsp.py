import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd

# Define ASEAN countries
cities_names = ["Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam"]

# Get user input for each city's coordinates
st.title("ASEAN TSP with User-defined Coordinates")

city_coords = {}
for city in cities_names:
    x_value = st.number_input(f"Enter x-coordinate for {city}", min_value=0, max_value=100, value=10, step=1)
    y_value = st.number_input(f"Enter y-coordinate for {city}", min_value=0, max_value=100, value=10, step=1)
    city_coords[city] = (x_value, y_value)

# Population parameters
n_population = st.number_input("Enter population size", min_value=50, max_value=500, value=250, step=10)
n_generations = st.number_input("Enter number of generations", min_value=50, max_value=1000, value=200, step=50)
crossover_per = st.slider("Enter crossover percentage", min_value=0.0, max_value=1.0, value=0.8)
mutation_per = st.slider("Enter mutation percentage", min_value=0.0, max_value=1.0, value=0.2)

# Colors for visualization
colors = sns.color_palette("pastel", len(cities_names))

city_icons = {
    "Brunei": "♕",
    "Cambodia": "♖",
    "Indonesia": "♗",
    "Laos": "♘",
    "Malaysia": "♙",
    "Myanmar": "♔",
    "Philippines": "♚",
    "Singapore": "♛",
    "Thailand": "♜",
    "Vietnam": "♝"
}

fig, ax = plt.subplots()
ax.grid(False)

# Plot each city based on user-provided coordinates
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons[city]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')

    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

fig.set_size_inches(16, 12)
st.pyplot(fig)

# Remaining code for genetic algorithm
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(0, len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = []
    for i in range(0, len(population)):
        total_dist_all_individuals.append(total_dist_individual(population[i]))
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - total_dist_all_individuals
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    return population_fitness_probs

def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
    return population[selected_individual_index]

def crossover(parent_1, parent_2):
    n_cities_cut = len(cities_names) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = parent_1[0:cut] + [city for city in parent_2 if city not in parent_1[0:cut]]
    offspring_2 = parent_2[0:cut] + [city for city in parent_1 if city not in parent_2[0:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    n_cities_cut = len(cities_names) - 1
    index_1 = round(random.uniform(0, n_cities_cut))
    index_2 = round(random.uniform(0, n_cities_cut))
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    for generation in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
        offspring = []
        for i in range(0, len(parents), 2):
            offspring_1, offspring_2 = crossover(parents[i], parents[i + 1])
            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)
            offspring.extend([offspring_1, offspring_2])
        population = parents + offspring
    return population

best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
shortest_path = min(best_mixed_offspring, key=total_dist_individual)

# Plot shortest path
x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i, txt in enumerate(shortest_path):
    ax.annotate(f"{i+1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

plt.title(f"TSP Best Route Using GA (Distance: {total_dist_individual(shortest_path):.2f})", fontsize=18)
fig.set_size_inches(16, 12)
st.pyplot(fig)
