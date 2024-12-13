import streamlit as st
import csv
import random
import pandas as pd

st.set_page_config(
    page_title="TV Ratings by Raja Izzudin"
)

st.header("TV Ratings by Raja Izzudin", divider="gray")

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)

        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings

    return program_ratings

# Path to the CSV file
file_path = 'program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

##################################### DEFINING PARAMETERS AND DATASET ################################################################
ratings = program_ratings_dict
GEN = 100
POP = 50
EL_S = 2  # elitism size

all_programs = list(ratings.keys())  # all programs
all_time_slots = list(range(6, 24))  # time slots

######################################### DEFINING FUNCTIONS ########################################################################
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=0.8, mutation_rate=0.2, elitism_size=EL_S):
    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

##################################################### MAIN LOGIC ###################################################################################
# User inputs for mutation and crossover rate ranges
crossover_rate_min = st.number_input("Minimum Crossover Rate", min_value=0.0, max_value=0.9, value=0.6, step=0.1)
crossover_rate_max = st.number_input("Maximum Crossover Rate", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
mutation_rate_min = st.number_input("Minimum Mutation Rate", min_value=0.01, max_value=0.1, value=0.02, step=0.01)
mutation_rate_max = st.number_input("Maximum Mutation Rate", min_value=0.02, max_value=0.2, value=0.05, step=0.01)

# Execute button
if st.button("Run Simulation"):
    results = []

    # Iterating over ranges of crossover and mutation rates
    for crossover_rate in [round(x, 2) for x in list(pd.np.arange(crossover_rate_min, crossover_rate_max + 0.1, 0.1))]:
        for mutation_rate in [round(x, 2) for x in list(pd.np.arange(mutation_rate_min, mutation_rate_max + 0.01, 0.01))]:

            # brute force
            all_possible_schedules = initialize_pop(all_programs, all_time_slots)
            initial_best_schedule = finding_best_schedule(all_possible_schedules)

            rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
            genetic_schedule = genetic_algorithm(
                initial_best_schedule,
                generations=GEN,
                population_size=POP,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                elitism_size=EL_S
            )

            final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

            total_ratings = fitness_function(final_schedule)

            results.append({
                "Crossover Rate": crossover_rate,
                "Mutation Rate": mutation_rate,
                "Total Ratings": total_ratings
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display the results
    st.subheader("TV Ratings based on Mutation and Crossover Rates")
    st.write(results_df)

    st.write("Best Configuration:")
    best_result = results_df.loc[results_df['Total Ratings'].idxmax()]
    st.write(best_result)
