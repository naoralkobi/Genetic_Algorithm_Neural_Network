# Genetic Neural Network Algorithm

This repository contains a Python implementation of a Genetic Algorithm for optimizing a neural network. The algorithm is designed to evolve and optimize the weights of a neural network using a genetic approach.

## Installation
1. Clone the repository:

```
gh repo clone naoralkobi/Genetic_Algorithm_Neural_Network
```

2. Change to the project directory:

```
cd genetic-neural-network
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage
1. Make sure you have a properly formatted input file nn1.txt in the project directory. The file should contain the training data with examples and labels, separated by whitespace.

2. Open the genetic_neural_network.py file and modify the parameters in the consts.py file according to your requirements.

3. Run the script:

```
python3 buildnet0.py
```

The script will train the neural network using the genetic algorithm and display the accuracy of the network on the test data.

## File Structure
The repository consists of the following files:

* **genetic_neural_network.py**: The main Python script that implements the genetic algorithm and neural network optimization.
* **activaction_functions.py**: Python module containing activation functions for the neural network.
* **consts.py**: Python module containing constants and parameters for the algorithm.
* **README.md**: Documentation file providing information about the repository and usage instructions.
* **nn0.txt**: First sample input file containing the training data.
* **nn1.txt**: Second sample input file containing the training data.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgments
This implementation is inspired by the concepts of genetic algorithms and neural networks. Special thanks to the authors and contributors of the libraries and resources used in this project.