import matplotlib.pyplot as plt
from itertools import permutations
from random import shuffle
import random
import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd

# Original coordinates for cities
x = [0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5]
y = [1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]
cities_names = ["Gliwice", "Cairo", "Rome", "Krakow", "Paris", "Alexandria", "Berlin", "Tokyo", "Rio", "Budapest"]
city_coords = dict(zip(cities_names, zip(x, y)))

# Streamlit title
st.title("TSP with Genetic Algorithm - Customize City Coordinates")

# Input fields for user to update coordinates
for city in cities_names:
    st.subheader(f"Set Coordinates for {city}")
    new_x = st.number_input(f"X-coordinate for {city}", min_value=-20, max_value=20, value=int(city_coords[city][0]))
    new_y = st.number_input(f"Y-coordinate for {city}", min_value=-20, max_value=20, value=int(city_coords[city][1]))
    city_coords[city] = (new_x, new_y)

# Parameters for genetic algorithm
n_population = 250
n_generations = 200
crossover_per = 0.8
mutation_per = 0.2

# Pastel colors and icons for visualization
colors = sns.color_palette("pastel", len(cities_names))
city_icons = {
    "Gliwice": "♕", "Cairo": "♖", "Rome": "♗", "Krakow": "♘", "Paris": "♙",
    "Alexandria": "♔", "Berlin": "♚", "Tokyo": "♛", "Rio": "♜", "Budapest": "♝"
}

# Visualization: Display cities and connections
fig, ax = plt.subplots()
ax.grid(False)
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

# Genetic Algorithm and TSP functions here...
# (Add genetic algorithm functions like initial_population, dist_two_cities, etc., as in the original code)

# Find best route and plot the shortest path
best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
shortest_path = min(best_mixed_offspring, key=total_dist_individual)

# Extract coordinates for shortest path plot
x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

# Visualization of the best route
fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()
plt.title(f"TSP Best Route Using GA (Distance: {total_dist_individual(shortest_path):.2f})", fontsize=18)
for i, txt in enumerate(shortest_path):
    ax.annotate(f"{i+1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)
fig.set_size_inches(16, 12)
st.pyplot(fig)
