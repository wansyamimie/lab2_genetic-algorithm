import random
import matplotlib.pyplot as plt
import streamlit as st

# Parameters given in the lab question
POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
GENERATIONS = 50


# Create one chromosome with 80 random bits (0 or 1)
def create_chromosome():
    chromosome = []
    for i in range(CHROMOSOME_LENGTH):
        chromosome.append(random.randint(0, 1))
    return chromosome


# Calculate the fitness of a chromosome
def calculate_fitness(chromosome):
    number_of_ones = sum(chromosome)
    fitness = 80 - abs(number_of_ones - TARGET_ONES)
    return fitness


# Create the starting population
def create_population():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(create_chromosome())
    return population


# Select the better chromosome from two random chromosomes
def selection(population):
    parent1 = random.choice(population)
    parent2 = random.choice(population)

    if calculate_fitness(parent1) > calculate_fitness(parent2):
        return parent1
    else:
        return parent2


# Combine two parents using one-point crossover
def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


# Change some bits from 0 to 1 or 1 to 0
def mutation(chromosome, mutation_rate):
    child = chromosome.copy()

    for i in range(CHROMOSOME_LENGTH):
        if random.random() < mutation_rate:
            if child[i] == 0:
                child[i] = 1
            else:
                child[i] = 0

    return child


# Run the genetic algorithm
def run_genetic_algorithm(mutation_rate):
    random.seed(42)
    population = create_population()
    best_fitness_history = []
    average_fitness_history = []

    for generation in range(GENERATIONS):
        fitness_values = []
        for chromosome in population:
            fitness_values.append(calculate_fitness(chromosome))

        best_fitness = max(fitness_values)
        average_fitness = sum(fitness_values) / POPULATION_SIZE
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)

        # Keep the best chromosome for the next generation
        best_index = fitness_values.index(best_fitness)
        best_chromosome = population[best_index].copy()
        new_population = [best_chromosome]

        # Create new chromosomes until the population has 300 members
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)

            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    # Find the best chromosome in the last generation
    fitness_values = []
    for chromosome in population:
        fitness_values.append(calculate_fitness(chromosome))

    best_fitness = max(fitness_values)
    best_index = fitness_values.index(best_fitness)
    best_chromosome = population[best_index]

    return best_chromosome, best_fitness_history, average_fitness_history


# Streamlit page
st.title("Genetic Algorithm - Lab Report 2")
st.write("This program uses a Genetic Algorithm to find a chromosome with 50 ones.")

st.subheader("Parameters")
st.write("Population size: 300")
st.write("Chromosome length: 80")
st.write("Target number of ones: 50")
st.write("Number of generations: 50")

mutation_rate = st.slider("Mutation rate", 0.001, 0.100, 0.010, 0.001)

if st.button("Run Genetic Algorithm"):
    best_chromosome, best_history, average_history = run_genetic_algorithm(mutation_rate)

    number_of_ones = sum(best_chromosome)
    number_of_zeros = CHROMOSOME_LENGTH - number_of_ones
    final_fitness = calculate_fitness(best_chromosome)

    st.subheader("Result")
    st.write("Best chromosome:")
    st.code("".join(str(bit) for bit in best_chromosome))
    st.write("Number of ones:", number_of_ones)
    st.write("Number of zeros:", number_of_zeros)
    st.write("Best fitness:", final_fitness, "/ 80")

    # Display the fitness for each generation
    st.subheader("Fitness Graph")
    fig, ax = plt.subplots()
    ax.plot(range(1, GENERATIONS + 1), best_history, label="Best fitness")
    ax.plot(range(1, GENERATIONS + 1), average_history, label="Average fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness over 50 Generations")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    if final_fitness == 80:
        st.success("The chromosome has exactly 50 ones. Maximum fitness is 80.")
    else:
        st.warning("This run did not reach the maximum fitness. Try running it again.")

st.subheader("How the program works")
st.write("1. Create 300 random chromosomes, each with 80 bits.")
st.write("2. Calculate fitness. A chromosome with 50 ones gets fitness 80.")
st.write("3. Select parents with higher fitness.")
st.write("4. Use crossover to exchange parts of the parents.")
st.write("5. Use mutation to change some bits.")
st.write("6. Repeat the process for 50 generations.")
