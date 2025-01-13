import numpy as np
import pandas as pd
import time
import streamlit as st


# Load and preprocess the insurance dataset
file_path = 'insurance.csv'
data = pd.read_csv(file_path)

# Encode categorical variables ('sex', 'smoker', 'region')
label_encoders = {
    'sex': LabelEncoder(),
    'smoker': LabelEncoder(),
    'region': LabelEncoder()
}

for column in ['sex', 'smoker', 'region']:
    data[column] = label_encoders[column].fit_transform(data[column])

# Split data into features (X) and target (y)
X = data.drop('charges', axis=1)
y = data['charges']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the fitness function for PSO
def fitness_function(params):
    # Initialize the linear regression model with parameters from PSO
    model = LinearRegression()
    model.coef_ = np.array(params[:-1])  # All except the last parameter for the coefficients
    model.intercept_ = params[-1]  # The last parameter for the intercept

    # Train the model and make predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Initialize particles for PSO
def initialize_particles(pop_size, dimensions, lower_bound, upper_bound):
    particles = np.random.uniform(lower_bound, upper_bound, (pop_size, dimensions))
    velocities = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), (pop_size, dimensions))
    return particles, velocities

# Update velocity
def update_velocity(velocities, particles, personal_best_positions, global_best_position, inertia, cognitive, social):
    r1 = np.random.random(size=particles.shape)
    r2 = np.random.random(size=particles.shape)

    cognitive_component = cognitive * r1 * (personal_best_positions - particles)
    social_component = social * r2 * (global_best_position - particles)
    velocities = inertia * velocities + cognitive_component + social_component
    return velocities

# Update particle positions
def update_position(particles, velocities, lower_bound, upper_bound):
    particles += velocities
    particles = np.clip(particles, lower_bound, upper_bound)
    return particles

# PSO optimization algorithm
def particle_swarm_optimization(pop_size, dimensions, lower_bound, upper_bound, max_generations, inertia, cognitive, social):
    particles, velocities = initialize_particles(pop_size, dimensions, lower_bound, upper_bound)
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_function(p) for p in particles])

    global_best_position = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    history = []
    start_time = time.time()  # Start time for computational efficiency

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

    # Calculate the computational time
    elapsed_time = time.time() - start_time
    return global_best_position, global_best_score, history, elapsed_time

# Streamlit Interface
st.title("Insurance Charges Prediction with Particle Swarm Optimization (PSO)")

# Parameters
pop_size = st.number_input("Population Size", min_value=10, value=50)
dimensions = st.number_input("Dimensions (Number of Features)", min_value=2, value=X_train.shape[1] + 1)  # +1 for the intercept term
max_generations = st.number_input("Maximum Generations", min_value=10, value=100)
lower_bound = st.number_input("Lower Bound", value=-1.0)
upper_bound = st.number_input("Upper Bound", value=1.0)
inertia = st.number_input("Inertia Weight", value=0.7)
cognitive = st.number_input("Cognitive Coefficient", value=1.5)
social = st.number_input("Social Coefficient", value=1.5)

if st.button("Run Optimization"):
    best_solution, best_fitness, history, elapsed_time = particle_swarm_optimization(
        pop_size, dimensions, lower_bound, upper_bound, max_generations, inertia, cognitive, social
    )
    st.success(f"Optimization Completed! Best Fitness (MSE): {best_fitness}")
    st.write("Best Solution (Model Parameters):", best_solution)

    # Plot convergence
    st.line_chart(history)

    # Display the computational efficiency (time taken)
    st.write(f"Computational Efficiency: Time taken = {elapsed_time:.2f} seconds")
