"""
Script to load a test dataset, use a pre-trained neural network for predictions, and save the results to a file.
"""

import pandas as pd
from buildnet0 import NeuralNetwork  # Assuming `buildnet0` contains the definition of the NeuralNetwork class
import numpy as np
import pickle
from utils import arrange_data

if __name__ == "__main__":
    # Load test dataset
    x_test = arrange_data("testnet0.txt")

    # Load pre-trained neural network
    with open('wnet0.pkl', 'rb') as f:
        best_network_result = pickle.load(f)

    # Make predictions using the loaded network
    predictions = best_network_result.predict(x_test)

    # Save predictions to a file
    with open("result0.txt", "w") as file:
        for label in predictions:
            file.write(str(label) + "\n")
