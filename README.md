# Computation Evolution

## Overview

Computation Evolution is a Python-based framework for evolving neural network architectures using an evolutionary algorithm. The core idea is to use adaptive computation layers that can mutate and evolve, allowing for the discovery of novel performance-increasing computations beyond high-level architectural changes.

## Features

- **AdaptiveComputationLayer**: A customizable layer that supports various operations and can evolve by adding, modifying, or removing operations.
- **CustomModel**: A neural network model incorporating adaptive computation layers.
- **Evolutionary Training**: Includes both synchronous and asynchronous training modes to evolve a population of neural networks.
- **Mutation and Crossover**: Implements mechanisms for mutating the network structure and combining parents to generate new models.

## Installation

To use Computation Evolution, you need to have Python 3.6 or later installed. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run the evolutionary training process, you can execute the `main.py` script with various command-line arguments:

```bash
python main.py --batch_size 32 --learning_rate 0.001 --epochs 5 --window_size 10 \
               --population_size 100 --generations 1000 --retain_top 0.5 \
               --random_select 0.1 --mutate_chance 0.3 --save_every 1 \
               --async_evolution True
```

### Arguments

- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for the optimizer.
- `--epochs`: Number of epochs for each evaluation.
- `--window_size`: Window size for time-series data.
- `--population_size`: Size of the population of models.
- `--generations`: Number of generations to evolve.
- `--retain_top`: Fraction of top-performing models to retain.
- `--random_select`: Probability of randomly selecting non-top models.
- `--mutate_chance`: Probability of mutating a model.
- `--save_every`: Interval of generations to save the best model.
- `--async_evolution`: Whether to use asynchronous evolution.

### Customization

You can customize the behavior of the `AdaptiveComputationLayer` by adding different types of operations, activations, or custom functions. The `evaluation_function` can also be modified to tailor the scoring criteria for evolved models.

## Example

```python
# Example usage
trainer = EvolutionaryTrainer(population_size=100, input_dim=10, output_dim=1)
evaluation_function = lambda network: evaluation_function(network, train_loader, val_loader, 5, 0.001, 32, 'cuda')
trainer.evolve(generations=1000, evaluation_function=evaluation_function, retain_top=0.5, random_select=0.1, mutate_chance=0.3)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
