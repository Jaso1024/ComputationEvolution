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

def worker_function(trainer, index, evaluation_function, mutate_chance, tournament_size):
    try:
        network = trainer.population[index]
        score = trainer._cached_evaluation(index, network, evaluation_function)
        trainer.scores[index] = score

        if random.random() < mutate_chance:
            network.adaptive_layer.mutate()
        else:
            with trainer.lock:
                tournament = random.sample(range(trainer.population_size), tournament_size)
            parent1 = max(tournament, key=lambda i: trainer.scores[i])
            parent2 = max(tournament, key=lambda i: trainer.scores[i] if i != parent1 else float('-inf'))
            child = trainer._crossover(trainer.population[parent1], trainer.population[parent2])
            trainer.population[index] = child

        return score
    except Exception as e:
        print(f"Error in worker: {e}")
        return float('-inf')

def evaluation_function(network, train_loader, val_loader, epochs, learning_rate, batch_size, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    network.train()
    network = network.to(device)
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(network.device), targets.to(network.device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    network.eval()
    network = network.to(device)
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(network.device), targets.to(network.device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    sum_loss = avg_loss + len(network.adaptive_layer.operations) * 0.05
    return -sum_loss
