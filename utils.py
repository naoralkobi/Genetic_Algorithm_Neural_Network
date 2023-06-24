import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def row_to_list(row: str):
    return list(row[0])

def create_png(generations, accuracy):
    """
    Creates a line plot of accuracy over generations and saves it as a PNG file.

    Parameters:
        generations (list): A list of integers representing the generations.
        accuracy (list): A list of floats representing the corresponding accuracy values.

    Returns:
        None

    Example:
        generations = [1, 2, 3, 4, 5]
        accuracy = [0.8, 0.85, 0.9, 0.95, 1.0]
        create_png(generations, accuracy)

    This function creates a line plot showing the change in accuracy over generations. It takes the list of generations
    and the corresponding accuracy values as input. The plot is saved as a PNG file named "fitness_scores.png".

    The function uses matplotlib library to create the plot. It creates a figure and axis, sets the size of the figure,
    plots the accuracy values against generations, and sets the labels and title of the plot. Finally, it saves the plot
    as a PNG file.

    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the size of the figure
    fig.set_size_inches(10, 5)

    # Plot generations against accuracy
    ax.plot(generations, accuracy)

    # Set labels and title
    ax.set_xlabel("Generations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Number of Fitness Score Over Time")

    # Save the plot as a PNG file
    plt.savefig("fitness_scores.png")


def split_data(train_path, test_path):
    """
    Splits the data into training and testing sets.

    Returns:
        training_set (numpy.ndarray): Examples for training.
        training_labels (numpy.ndarray): Labels for training.
        test_set (numpy.ndarray): Examples for testing.
        test_labels (numpy.ndarray): Labels for testing.
    """
    # for train
    read_train = pd.read_csv(train_path, sep='\s+', header=None, dtype=str)
    training_examples = read_train.iloc[:, 0].astype(str)
    transpose = training_examples.to_frame()
    train_matrix = transpose.apply(row_to_list, axis='columns', result_type='expand')
    train_matrix = train_matrix.to_numpy(int)
    training_labels = read_train.iloc[:, 1].values.astype(int)
    # for test
    read_test = pd.read_csv(test_path, sep='\s+', header=None, dtype=str)
    test_examples = read_test.iloc[:, 0].astype(str)
    transpose = test_examples.to_frame()
    test_matrix = transpose.apply(row_to_list, axis='columns', result_type='expand')
    test_matrix = test_matrix.to_numpy(int)
    test_labels = read_test.iloc[:, 1].values.astype(int)


    training_examples = train_matrix[:train_matrix.shape[0]]
    test_examples = test_matrix[:test_matrix.shape[0]]
    return training_examples, training_labels, test_examples, test_labels





def calculate_accuracy(train_labels, predictions):
    """
    Calculates the accuracy of predictions compared to the true labels.

    Args:
        train_labels (numpy.ndarray): True labels of the training examples.
        predictions (numpy.ndarray): Predicted labels for the training examples.

    Returns:
        accuracy (float): The accuracy of the predictions.
    """
    # Compare predictions to true labels and calculate the accuracy
    correct_predictions = predictions == train_labels
    return np.mean(correct_predictions)




def write_wnet(network, file_path):
    """
    Write a network object to a file using pickle serialization.

    Args:
        network: The network object to be serialized and saved.

    Returns:
        None
    """

    # Open the file in write binary mode
    with open(file_path, 'wb') as file:
        # Serialize and dump the network object to the file
        pickle.dump(network, file)

