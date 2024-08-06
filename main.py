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
from utils.functions import worker_function, evaluation_function
from utils.AsynchronousEvolutionaryTrainer import AsynchronousEvolutionaryTrainer
from utils.SynchronousEvolutionaryTrainer import SynchronousEvolutionaryTrainer


def main(args):
    global train_loader, val_loader

    np.random.seed(42)
    timesteps = 1000
    features = 1
    data_series = np.random.randn(timesteps, features)
    series = np.sum(data_series, axis=1) + np.random.randn(timesteps) * 0.3

    def create_supervised_dataset(data_series, series, window_size):
        X, y = [], []
        for i in range(len(data_series) - window_size):
            X.append(data_series[i:i + window_size])
            y.append(series[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_supervised_dataset(data_series, series, args.window_size)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = data.TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    output_dim = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.async_evolution:
        trainer = AsynchronousEvolutionaryTrainer(args.population_size, input_dim, output_dim, device)
    
        eval_func = lambda network: evaluation_function(network, train_loader, val_loader, 
                                                        args.epochs, args.learning_rate, 
                                                        args.batch_size, device)
        
        trainer.evolve_async(num_evaluations=args.generations * args.population_size, 
                            evaluation_function=eval_func,
                            mutate_chance=args.mutate_chance)
    else:
        trainer = SynchronousEvolutionaryTrainer(args.population_size, input_dim, output_dim, device)
        
        eval_func = lambda network: evaluation_function(network, train_loader, val_loader, 
                                                        args.epochs, args.learning_rate, 
                                                        args.batch_size, device)
        
        trainer.evolve(generations=args.generations, evaluation_function=eval_func,
                    retain_top=args.retain_top, random_select=args.random_select,
                    mutate_chance=args.mutate_chance, save_every=args.save_every)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutionary Neural Network Trainer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for each evaluation")
    parser.add_argument("--window_size", type=int, default=10, help="Window size for time series data")
    parser.add_argument("--population_size", type=int, default=100, help="Size of the population")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations to evolve")
    parser.add_argument("--retain_top", type=float, default=0.5, help="Fraction of top performers to retain")
    parser.add_argument("--random_select", type=float, default=0.1, help="Probability of random selection")
    parser.add_argument("--mutate_chance", type=float, default=0.3, help="Probability of mutation")
    parser.add_argument("--save_every", type=int, default=1, help="Save best model every n generations")
    parser.add_argument("--async_evolution", type=bool, default=True, help="Evolve asynchronously")
    args = parser.parse_args()
    main(args)