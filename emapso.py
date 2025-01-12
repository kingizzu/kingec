import numpy as np
import streamlit as st

def fitness_function(solution):
    return sum(x**2 for x in solution)

def initialize_particles(pop_size, dimensions, lower_bound, upper_bound):
    particles = np.random.uniform(lower_bound, upper_bound, (pop_size, dimensions))
    velocities = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), (pop_size, dimensions))
    return particles, velocities

def update_velocity(velocities, particles, personal_best_positions, global_best_position, inertia, cognitive, social):
    r1 = np.random.random(size=particles.shape)
    r2 = np.random.random(size=particles.shape)

    cognitive_component = cognitive * r1 * (personal_best_positions - particles)
    social_component = social * r2 * (global_best_position - particles)
    velocities = inertia * velocities + cognitive_component + social_component
    return velocities

def update_position(particles, velocities, lower_bound, upper_bound):
    particles += velocities
    particles = np.clip(particles, lower_bound, upper_bound)
    return particles

def particle_swarm_optimization(pop_size, dimensions, lower_bound, upper_bound, max_generations, inertia, cognitive, social):
    particles, velocities = initialize_particles(pop_size, dimensions, lower_bound, upper_bound)
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_function(p) for p in particles])

    global_best_position = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    history = []

    for generation in range(max_generations):
        velocities = update_velocity(velocities, particles, personal_best_positions, global_best_position, inertia, cognitive, social)
        particles = update_position(particles, velocities, lower_bound, upper_bound)

        current_scores = np.array([fitness_function(p) for p in particles])

        # Update personal bests
        for i in range(pop_size):
            if current_scores[i] < personal_best_scores[i]:
                personal_best_scores[i] = current_scores[i]
                personal_best_positions[i] = particles[i]

        # Update global best
        if np.min(current_scores) < global_best_score:
            global_best_score = np.min(current_scores)
            global_best_position = particles[np.argmin(current_scores)]

        history.append(global_best_score)

        if generation % 10 == 0 or generation == max_generations - 1:
            st.write(f"Generation {generation}: Best Fitness = {global_best_score}")

    return global_best_position, global_best_score, history

# Streamlit Interface
st.title("Hybrid Exchange Market using Particle Swarm Optimization (PSO)")

# Parameters
pop_size = st.number_input("Population Size", min_value=10, value=50)
dimensions = st.number_input("Dimensions", min_value=2, value=10)
max_generations = st.number_input("Maximum Generations", min_value=10, value=100)
lower_bound = st.number_input("Lower Bound", value=-1.0)
upper_bound = st.number_input("Upper Bound", value=1.0)
inertia = st.number_input("Inertia Weight", value=0.7)
cognitive = st.number_input("Cognitive Coefficient", value=1.5)
social = st.number_input("Social Coefficient", value=1.5)

if st.button("Run Optimization"):
    best_solution, best_fitness, history = particle_swarm_optimization(
        pop_size, dimensions, lower_bound, upper_bound, max_generations, inertia, cognitive, social
    )
    st.success(f"Optimization Completed! Best Fitness: {best_fitness}")
    st.write("Best Solution:", best_solution)

    # Plot convergence
    st.line_chart(history)
