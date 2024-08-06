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



class SynchronousEvolutionaryTrainer:
    def __init__(self, population_size, input_dim, output_dim, device='cuda'):
        self.population_size = population_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.population = [self._create_random_network().to(self.device) for _ in range(population_size)]
        self.cache = {}

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


    def _create_random_network(self):
        return CustomModel(self.input_dim, self.output_dim)

    def evolve(self, generations, evaluation_function, retain_top=0.5, random_select=0.1, mutate_chance=0.3, save_every=1):
        bar = tqdm(range(generations), desc="Evolving Generations")
        for generation in bar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                scores = list(executor.map(lambda n: self._cached_evaluation(n, evaluation_function), self.population))
            

            avg_score = sum(scores) / len(scores)
            top_scores = sorted(scores, reverse=True)[:10]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:int(len(scores) * retain_top)]
            parents = [self.population[i] for i in top_indices]

            for i in range(len(scores)):
                if random_select > random.random() and i not in top_indices:
                    parents.append(self.population[i])

            for individual in parents:
                if mutate_chance > random.random():
                    individual.adaptive_layer.mutate()

            self.population = self._generate_new_population(parents)

            bar.set_description(f"Best Score = {scores[top_indices[0]]:.4f} | Avg Score = {avg_score:.4f}")

            if generation % save_every == 0 and generation > 2:
                self.visualize_best(generation, self.population[top_indices[0]])

    def visualize_best(self, generation, best_model):
        visualization = best_model.adaptive_layer.visualize_operations()
        print(visualization)
        file_name = "Best Computation Order.txt"
        with open(file_name, "w") as file:
            file.write(visualization = best_model.adaptive_layer.visualize_operations())

    def _generate_new_population(self, parents):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self._crossover(parent1, parent2)
            new_population.append(child)
        return new_population

    def _crossover(self, parent1, parent2):
        child = self._create_random_network()
        for layer_name, child_layer in child.named_modules():
            if isinstance(child_layer, AdaptiveComputationLayer):
                parent_layer = random.choice([parent1, parent2]).adaptive_layer
                child_layer.operations = copy.deepcopy(parent_layer.operations.copy())
                child_layer.params = copy.deepcopy(nn.ParameterDict({name: param.clone() for name, param in parent_layer.params.items()}))
                child_layer.nodes = copy.deepcopy(parent_layer.nodes)
                child_layer.edges = copy.deepcopy(parent_layer.edges)
        return child

    def _save_population(self, generation):
        with open(f"population_gen_{generation}.pkl", "wb") as f:
            pickle.dump(self.population, f)
