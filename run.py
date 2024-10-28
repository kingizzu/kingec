import streamlit as st
import random

# Function to calculate fitness
def calculate_fitness(target, test_string):
    fitness = sum(1 for expected, actual in zip(target, test_string) if expected == actual)
    return fitness

# Function to mutate a string
def mutate_string(test_string, mutation_rate):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new_string = []
    for char in test_string:
        if random.random() < mutation_rate:
            new_string.append(random.choice(alphabet))
        else:
            new_string.append(char)
    return new_string

# Streamlit app
st.title("Genetic Algorithm")

# User input
name = st.text_input("Enter your name", "")
mutation_rate = st.slider("Enter your mutation rate", 0.0, 1.0, 0.1)
calculate = st.button("Calculate")

if calculate and name:
    target = list(name)
    test_string = [random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(len(target))]
    generation = 0

    # Run the genetic algorithm until the target is found
    while True:
        generation += 1
        fitness = calculate_fitness(target, test_string)
        
        # Display the current string, generation, and fitness
        st.write(f"String: {test_string} Generation: {generation} Fitness: {fitness}")
        
        # Check if the target is found
        if fitness == len(target):
            st.write("Target found")
            break

        # Mutate the string
        test_string = mutate_string(test_string, mutation_rate)
