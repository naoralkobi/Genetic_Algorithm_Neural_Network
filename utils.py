import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from consts import test_ratio


def row_to_list(row: str):
    return list(row[0])


def create_png(generations, accuracy, file_name):
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
    :param accuracy:
    :param generations:
    :param file_name:

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
    plt.savefig(file_name)


# def split_data(train_path, test_path):
#     """
#     Splits the data into training and testing sets.
#
#     Returns:
#         training_set (numpy.ndarray): Examples for training.
#         training_labels (numpy.ndarray): Labels for training.
#         test_set (numpy.ndarray): Examples for testing.
#         test_labels (numpy.ndarray): Labels for testing.
#     """
#     # for train
#     read_train = pd.read_csv(train_path, sep='\s+', header=None, dtype=str)
#     training_examples = read_train.iloc[:, 0].astype(str)
#     transpose = training_examples.to_frame()
#     train_matrix = transpose.apply(row_to_list, axis='columns', result_type='expand')
#     train_matrix = train_matrix.to_numpy(int)
#     training_labels = read_train.iloc[:, 1].values.astype(int)
#     # for test
#     read_test = pd.read_csv(test_path, sep='\s+', header=None, dtype=str)
#     test_examples = read_test.iloc[:, 0].astype(str)
#     transpose = test_examples.to_frame()
#     test_matrix = transpose.apply(row_to_list, axis='columns', result_type='expand')
#     test_matrix = test_matrix.to_numpy(int)
#     test_labels = read_test.iloc[:, 1].values.astype(int)
#
#     training_examples = train_matrix[:train_matrix.shape[0]]
#     test_examples = test_matrix[:test_matrix.shape[0]]
#     return training_examples, training_labels, test_examples, test_labels


def split_data(file_path='nn0.txt'):
    """
    Splits the data into training and testing sets.

    Returns:
        training_set (numpy.ndarray): Examples for training.
        training_labels (numpy.ndarray): Labels for training.
        test_set (numpy.ndarray): Examples for testing.
        test_labels (numpy.ndarray): Labels for testing.
    """
    # Read data from the file
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, dtype=str)

    # Extract examples and labels from the data
    examples = data.iloc[:, 0].astype(str)
    transpose = examples.to_frame()
    matrix = transpose.apply(row_to_list, axis='columns', result_type='expand')
    matrix = matrix.to_numpy(int)
    labels = data.iloc[:, 1].values.astype(int)

    # Calculate the number of examples for testing
    train_len = int(test_ratio * matrix.shape[0])

    # Shuffle the indices
    indices = np.arange(matrix.shape[0])
    np.random.shuffle(indices)

    # Split the data into training and testing sets based on the indices
    training_idx, test_idx = indices[:train_len], indices[train_len:]
    test_set, training_set = matrix[training_idx], matrix[test_idx]
    test_labels, training_labels = labels[training_idx], labels[test_idx]

    return training_set, training_labels, test_set, test_labels


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


def write_object_to_file(network, file_path):
    """
    Write a network object to a file using pickle serialization.

    Args:
        network: The network object to be serialized and saved.

    Returns:
        None
        :param file_path:
    """

    # Open the file in write binary mode
    with open(file_path, 'wb') as file:
        # Serialize and dump the network object to the file
        pickle.dump(network, file)


def arrange_data(filename):
    # Read data from the file
    data = pd.read_csv(filename, delim_whitespace=True, header=None, dtype=str)

    # Extract examples and labels from the data
    examples = data.iloc[:, 0].astype(str)
    transpose = examples.to_frame()
    matrix = transpose.apply(row_to_list, axis='columns', result_type='expand')
    test_set = matrix.to_numpy(int)

    return test_set

