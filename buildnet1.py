import copy
import random
import numpy as np
import time
import activaction_functions
from utils import *
from consts import *


def define_neural_network():
    """
    Initializes a neural network model with random weights.

    Returns:
        nn_model (Individual): Neural network model with randomly initialized weights.
    """
    neural_network = NeuralNetwork()
    neural_network.add_weights(input_size, layer1_size)
    neural_network.add_weights(layer1_size, layer2_size)
    neural_network.add_weights(layer2_size, output_size)
    return neural_network


def create_population():
    """
    Initializes the population of neural networks.
    """
    population = [define_neural_network() for _ in range(population_size)]
    return population


class GeneticAlgorithm:
    """
    This class represents a Genetic Algorithm for optimizing a neural network.
    """

    def __init__(self, file_path):
        """
        Initializes the genetic algorithm with a population and dataset.
        """
        self.population = create_population()
        # possible to pass path of text file to split_data.
        self.x_train, self.y_train, self.x_test, self.y_test = split_data(file_path)
        self.best_fitness = 0
        self.same_fitness_count = 0
        self.generation = 0
        self.generations = []
        self.list_of_accuracy = []

    def tournament_selection(self, population, tournament_size):
        """
        Performs Tournament Selection on the population.

        Args:
            population (list): The population from which individuals are selected.
            tournament_size (int): The number of individuals participating in each tournament.

        Returns:
            selected_individuals (list): The selected individuals after tournament selection.
        """
        # Calculate fitness values for each individual in the population
        fitness_values = [self.calculate_fitness(individual) for individual in population]

        # Perform tournament selection by selecting the best individual in each tournament
        selected_individuals = []
        for _ in range(population_size):
            tournament_participants = random.sample(range(population_size), tournament_size)
            best_individual = None
            best_fitness = float('-inf')
            for participant in tournament_participants:
                fitness = fitness_values[participant]
                if fitness > best_fitness:
                    best_individual = population[participant]
                    best_fitness = fitness
            selected_individuals.append(best_individual)

        return selected_individuals

    def rank_selection(self, population):
        """
        Performs Rank Selection on the population.

        Args:
            population (list): The population from which individuals are selected.

        Returns:
            selected_individuals (list): The selected individuals after rank selection.
        """
        # Calculate fitness values for each individual in the population
        fitness_values = [self.calculate_fitness(individual) for individual in population]

        # Perform rank selection by choosing individuals based on selection probabilities
        selected_individuals = []
        selection_probs = self.calculate_selection_probs()
        for _ in range(population_size):
            individual_idx = self.choose_individual(selection_probs)
            selected_individuals.append(population[individual_idx])

        return selected_individuals

    @staticmethod
    def calculate_selection_probs():
        """
        Calculates selection probabilities based on rank.

        Returns:
            selection_probs (list): The selection probabilities for rank selection.
        """
        selection_probs = [rank / population_size for rank in range(1, population_size + 1)]
        return selection_probs

    @staticmethod
    def choose_individual(selection_probs):
        """
        Chooses an individual index based on selection probabilities.

        Args:
            selection_probs (list): The selection probabilities for rank selection.

        Returns:
            individual_idx (int): Index of the chosen individual.
        """
        cumulative_probs = np.cumsum(selection_probs)
        rand_num = np.random.random()
        individual_idx = np.argmax(cumulative_probs >= rand_num)

        return individual_idx

    def selection(self, fitness_list):
        """
        Selects the top individuals based on fitness.

        Args:
            fitness_list (list): The fitness scores of the individuals in the population.

        Returns:
            top_individuals (list): The top individuals selected based on fitness.
            remaining_individuals (list): The remaining individuals in the population.
        """
        # Sort the indices of individuals based on fitness in descending order
        sorted_indices = np.argsort(fitness_list)[::-1]

        # Select the top individuals based on the selection rate and population size
        top_individuals = [self.population[i] for i in sorted_indices[:int(population_size * selection_rate)]]

        # Determine the remaining individuals by excluding the top individuals
        remaining_individuals = list(set(self.population) - set(top_individuals))

        return top_individuals, remaining_individuals

    @staticmethod
    def crossover(selected_parents, top_individuals):
        """
        Performs crossover on the parents to produce offspring.

        Args:
            selected_parents (list): The selected parents for crossover.
            top_individuals (list): The top individuals in the population.

        Returns:
            offspring (list): The offspring generated from crossover.
        """
        offspring = []

        # Perform crossover to produce offspring until reaching the desired population size
        for _ in range((population_size - len(top_individuals)) // 2):
            # Select two parents randomly from the selected parents and top individuals
            parent1 = np.random.choice(selected_parents)
            parent2 = np.random.choice(top_individuals)

            # Perform crossover operation between the selected parents
            offspring += parent1.crossover(parent2)

        return offspring

    def calculate_fitness(self, network):
        """
        Calculates the fitness of a network based on its accuracy in predicting the training data.

        Args:
            network (NeuralNetwork): The network for which the fitness is calculated.

        Returns:
            fitness (float): The fitness value indicating the accuracy of the network.
        """
        # Calculate the accuracy of the network's predictions on the training data
        predictions = network.predict(self.x_train)
        accuracy = calculate_accuracy(self.y_train, predictions)

        return accuracy

    def evolve(self):
        """
        Runs the genetic algorithm.

        Returns:
            best_individual (Individual): Best individual from the final population.
        """
        for i in range(generations_num):
            self.generation += 1
            self.generations.append(i + 1)

            # calculate the fitness for the population
            fitness_scores = [round(self.calculate_fitness(network), 5) for network in self.population]

            # save the highest fitness
            current_fitness = max(fitness_scores)
            self.list_of_accuracy.append(max(fitness_scores))
            # convergence
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.same_fitness_count = 0
                max_index = self.population[np.argmax(fitness_scores)]
            else:
                self.same_fitness_count += 1

            if self.same_fitness_count >= stop_condition:
                break

            print(f"Generation {i + 1} best fitness score: {self.best_fitness}")

            # Perform selection, crossover, and mutation
            top_individuals, remaining_individuals = self.selection(fitness_scores)
            selected_parents = self.rank_selection(remaining_individuals)
            offsprings = self.crossover(selected_parents, top_individuals)

            # Determine the number of individuals to be left untouched for the next generation
            num_untouched = int((population_size - top_individuals_size) * untouched_ratio_nn1)
            untouched_offsprings = offsprings[:num_untouched]

            # mutate the other offsprings, that can be touched
            for offspring in offsprings[num_untouched:]:
                offspring.mutation()

            # the creation of the next population
            self.population = top_individuals + untouched_offsprings + offsprings[num_untouched:]

            if self.same_fitness_count > lamarckian_condition:
                self.population = [self.lamarckian_modification(network) for network in self.population]

            min_index = np.argmin(fitness_scores)
            self.population[min_index] = max_index

        fitness_scores = [self.calculate_fitness(network) for network in self.population]
        best_ind = self.population[np.argmax(fitness_scores)]
        return best_ind

    def lamarckian_modification(self, network):
        """
        Implements Lamarckian inheritance for the genetic algorithm.

        Args:
            network (NeuralNetwork): The network to undergo Lamarckian inheritance.

        Returns:
            updated_network (Individual): The network after the Lamarckian inheritance process.
        """
        # Calculate the fitness of the original network
        old_fitness = self.calculate_fitness(network)

        # Create a deep copy of the original network
        new_network = copy.deepcopy(network)

        # Perform Lamarckian inheritance by modifying random weights in the network
        for i in range(limit_lamarckian):
            # Choose a random layer and weight in the layer
            layer_num = np.random.choice(len(new_network.layers))
            layer_weights = new_network.layers[layer_num]
            rand_idx = np.random.randint(0, layer_weights.shape[0])
            rand_weight = np.random.randn(layer_weights.shape[1])

            # Update the weight in the new network
            new_network.layers[layer_num][rand_idx] = rand_weight

        # Check if the updated network has a higher fitness
        updated_fitness = self.calculate_fitness(network)
        updated_network = network if updated_fitness > old_fitness else new_network

        return updated_network


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_weights(self, inside, outside):
        """
        Adds a weight matrix to the neural network model with random initialization.

        Args:
            inside (int): Number of input neurons for the weight matrix.
            outside (int): Number of output neurons for the weight matrix.
        """
        # Randomly initialize weight matrix with values drawn from a Gaussian distribution
        # scaled by the square root of 1/inside to help with weight initialization
        weight_matrix = np.random.randn(inside, outside) * np.sqrt(1 / inside)
        self.add_layer(weight_matrix)

    def forward(self, examples: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the neural network layers to obtain the final output.

        Args:
            examples (np.ndarray): Input examples for the forward pass.

        Returns:
            output (np.ndarray): Output of the neural network after the forward pass.
        """
        samples = examples
        for w in self.layers[:-1]:
            samples = np.dot(samples, w)
            samples = activaction_functions.sigmoid(samples)
        # last layer
        w = self.layers[-1]
        samples = np.dot(samples, w)
        samples = activaction_functions.leaky_relu(samples)
        return samples

    def predict(self, inputs):
        """
        Performs predictions using the neural network model.

        Args:
            inputs (numpy.ndarray): Input data for prediction.

        Returns:
            predictions (numpy.ndarray): Binary predictions for the input data.
        """
        # Forward pass to obtain the outputs of the neural network
        outputs = self.forward(inputs)

        # Convert outputs to binary predictions using a threshold of 0.5
        binary_predictions = (outputs > 0.5).astype(int)

        # Flatten the binary predictions array to obtain a 1-dimensional array
        predictions = binary_predictions.flatten()

        return predictions

    def crossover(self, other_network):
        """
        Performs crossover between two parent networks to produce two child individuals.

        Args:
            other_network (NeuralNetwork): The other parent network for crossover.

        Returns:
            child1 (Individual): The first child individual resulting from crossover.
            child2 (Individual): The second child individual resulting from crossover.
        """
        child1 = NeuralNetwork()
        child2 = NeuralNetwork()

        # Get the layers from the parents and perform crossover
        for w1, w2 in zip(self.layers, other_network.layers):
            # Randomly select columns for crossover
            columns = np.random.choice([True, False], size=w1.shape[1])

            # Perform crossover operation using the selected columns
            c1w = np.where(columns, w1, w2)
            c2w = np.where(columns, w2, w1)

            # Add the updated layers to the child individuals
            child1.layers.append(c1w)
            child2.layers.append(c2w)

        return child1, child2

    def mutation(self):
        """
        Applies mutation to the weights of a neural network individual.

        Randomly replaces a weight in a randomly chosen layer with a new random weight.
        """
        # Check if mutation should occur based on the mutation rate
        if np.random.random() < mutation_rate:
            # Choose a random layer from self.layers
            layer_num = np.random.choice(len(self.layers))

            # Choose a random weight in the layer
            layer_weights = self.layers[layer_num]
            rand_idx = np.random.randint(0, layer_weights.shape[0])
            rand_weight = np.random.randn(layer_weights.shape[1])

            # Replace the random weight with the new random weight
            self.layers[layer_num][rand_idx] = rand_weight


if __name__ == '__main__':
    # Enable raising exceptions for all floating-point errors
    np.seterr(all="raise")

    start_time = time.perf_counter()

    # Initialize the genetic algorithm
    ga = GeneticAlgorithm('nn1.txt')

    # Train the network
    best_network = ga.evolve()
    write_object_to_file(best_network, 'wnet1.pkl')

    # Test the network
    predict_test = best_network.predict(ga.x_test)
    accuracy = calculate_accuracy(ga.y_test, predict_test)
    print(f"accuracy: {accuracy}")

    # Visualize the progress
    create_png(ga.generations, ga.list_of_accuracy, "fitness_scores_nn1")

    end_time = time.perf_counter()
    total_time = (end_time - start_time) / 60
    print("running time:", total_time)
