import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm
import pickle
import random
import copy
import random
from collections import defaultdict
import concurrent.futures
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import hashlib
import torch.multiprocessing as mp
from .AdaptiveModel import CustomModel, AdaptiveComputationLayer

class AsynchronousEvolutionaryTrainer:
    def __init__(self, population_size, input_dim, output_dim, device='cuda', show_every_n=100):
        self.population_size = population_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.population = [self._create_random_network().to(self.device) for _ in range(population_size)]
        self.scores = [float('-inf')] * population_size
        self.cache = {}
        self.show_every_n = show_every_n

    def _create_random_network(self):
        return CustomModel(self.input_dim, self.output_dim)

    def _get_network_hash(self, network):
        hash_input = str(network.adaptive_layer.operations) + str(network.adaptive_layer.params)
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _cached_evaluation(self, network, evaluation_function):
        network_hash = self._get_network_hash(network)
        if network_hash in self.cache:
            return self.cache[network_hash]
        score = evaluation_function(network)
        self.cache[network_hash] = score
        return score
    
    def visualize_best(self, best_model):
        print(best_model.adaptive_layer.visualize_operations())

    def evolve_async(self, num_evaluations, evaluation_function, tournament_size=3, mutate_chance=0.3):
        pbar = tqdm(total=num_evaluations, desc="Evaluations")
        for evaluation in range(num_evaluations):
            index = random.randint(0, self.population_size - 1)
            score = self._cached_evaluation(self.population[index], evaluation_function)
            self.scores[index] = score

            if random.random() < mutate_chance:
                self.population[index].adaptive_layer.mutate()
            else:
                tournament = random.sample(range(self.population_size), tournament_size)
                parent1 = max(tournament, key=lambda i: self.scores[i])
                parent2 = max(tournament, key=lambda i: self.scores[i] if i != parent1 else float('-inf'))
                child = self._crossover(self.population[parent1], self.population[parent2])
                self.population[index] = child

            best_index = max(range(self.population_size), key=lambda i: self.scores[i])
            pbar.set_postfix({'Best Score': f"{self.scores[best_index]:.4f}"})
            pbar.update(1)

            if evaluation % self.show_every_n == 0 and evaluation > 1:
                self.visualize_best(self.population[best_index])


        pbar.close()

    def _crossover(self, parent1, parent2):
        child = self._create_random_network()
        for layer_name, child_layer in child.named_modules():
            if isinstance(child_layer, AdaptiveComputationLayer):
                parent_layer = random.choice([parent1, parent2]).adaptive_layer
                child_layer.operations = copy.deepcopy(parent_layer.operations)
                child_layer.params = nn.ParameterDict({name: param.clone() for name, param in parent_layer.params.items()})
                child_layer.nodes = copy.deepcopy(parent_layer.nodes)
                child_layer.edges = copy.deepcopy(parent_layer.edges)
        return child
