import pandas as pd

from buildnet0 import Individual
import numpy as np
import pickle


def load_test_data(filename):
    with open(filename, 'r') as test_file:
        lines = test_file.readlines()

    data = [[int(bit) for bit in line.strip()] for line in lines]
    return np.array(data)


def main():
    x_test = load_test_data("testnet0.txt")
    with open('wnet0.pkl', 'rb') as file:
        best_network = pickle.load(file)
    predictions_labels = best_network.predict(x_test)
    with open("result0.txt", "w") as file:
        for label in predictions_labels:
            file.write(str(label) + "\n")


if __name__ == "__main__":
    main()
