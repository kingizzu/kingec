# New Coordinates for ASEAN countries
x = [5, 11, 18, 14, 8, 10, 22, 9, 15, 17]  # Replace with actual x-coordinates if available
y = [10, 7, 15, 12, 6, 8, 20, 5, 13, 9]    # Replace with actual y-coordinates if available
cities_names = ["Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam"]

# Update city_coords dictionary
city_coords = dict(zip(cities_names, zip(x, y)))

# Update city_icons (optional)
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

# Update the colors palette for the number of ASEAN countries
colors = sns.color_palette("pastel", len(cities_names))

# Running the code with updated ASEAN country list and coordinates
best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

total_dist_all_individuals = []
for i in range(0, n_population):
    total_dist_all_individuals.append(total_dist_individual(best_mixed_offspring[i]))

index_minimum = np.argmin(total_dist_all_individuals)
minimum_distance = min(total_dist_all_individuals)
st.write(minimum_distance)

# Shortest path
shortest_path = best_mixed_offspring[index_minimum]
st.write(shortest_path)

x_shortest = []
y_shortest = []
for city in shortest_path:
    x_value, y_value = city_coords[city]
    x_shortest.append(x_value)
    y_shortest.append(y_value)

x_shortest.append(x_shortest[0])
y_shortest.append(y_shortest[0])

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i in range(len(x)):
    for j in range(i + 1, len(x)):
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

plt.title(label="TSP Best Route Using GA",
          fontsize=25,
          color="k")

str_params = '\n'+str(n_generations)+' Generations\n'+str(n_population)+' Population Size\n'+str(crossover_per)+' Crossover\n'+str(mutation_per)+' Mutation'
plt.suptitle("Total Distance Travelled: "+str(round(minimum_distance, 3))+str_params, fontsize=18, y=1.047)

for i, txt in enumerate(shortest_path):
    ax.annotate(str(i+1)+ "- " + txt, (x_shortest[i], y_shortest[i]), fontsize=20)

fig.set_size_inches(16, 12)
st.pyplot(fig)
