#!/usr/bin/env python3
"""
🧠 Neural Architecture Search for Trading V2.0
Automated neural network design with evolutionary algorithms
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NeuralBlock(nn.Module):
    """Modular neural block for architecture search"""
    
    def __init__(self, input_size, output_size, block_type='dense'):
        super().__init__()
        self.block_type = block_type
        
        if block_type == 'dense':
            self.layer = nn.Linear(input_size, output_size)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        elif block_type == 'residual':
            self.layer1 = nn.Linear(input_size, output_size)
            self.layer2 = nn.Linear(output_size, output_size)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
        elif block_type == 'attention':
            self.query = nn.Linear(input_size, output_size)
            self.key = nn.Linear(input_size, output_size)
            self.value = nn.Linear(input_size, output_size)
            self.softmax = nn.Softmax(dim=-1)
        elif block_type == 'lstm':
            self.lstm = nn.LSTM(input_size, output_size, batch_first=True)
            self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        if self.block_type == 'dense':
            return self.dropout(self.activation(self.layer(x)))
        elif self.block_type == 'residual':
            residual = x if x.shape[-1] == self.layer2.out_features else self.layer1(x)
            out = self.activation(self.layer1(x))
            out = self.layer2(out)
            return self.dropout(out + residual)
        elif self.block_type == 'attention':
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            attention = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1)))
            return torch.matmul(attention, v)
        elif self.block_type == 'lstm':
            out, _ = self.lstm(x.unsqueeze(1))
            return self.dropout(out.squeeze(1))


class EvolutionaryArchitecture(nn.Module):
    """Evolutionary neural architecture"""
    
    def __init__(self, input_size, output_size, genome):
        super().__init__()
        self.genome = genome
        self.blocks = nn.ModuleList()
        
        current_size = input_size
        
        for i, gene in enumerate(genome):
            block_type = gene['type']
            hidden_size = gene['size']
            
            if i == len(genome) - 1:  # Last layer
                hidden_size = output_size
            
            block = NeuralBlock(current_size, hidden_size, block_type)
            self.blocks.append(block)
            current_size = hidden_size
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ArchitectureEvolution:
    """Evolutionary algorithm for neural architecture search"""
    
    def __init__(self, input_size, output_size, population_size=20):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        
        # Architecture constraints
        self.max_layers = 8
        self.layer_sizes = [64, 128, 256, 512, 1024]
        self.block_types = ['dense', 'residual', 'attention', 'lstm']
        
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population"""
        self.population = []
        
        for _ in range(self.population_size):
            genome = self._create_random_genome()
            self.population.append(genome)
    
    def _create_random_genome(self):
        """Create random neural architecture genome"""
        num_layers = random.randint(3, self.max_layers)
        genome = []
        
        for i in range(num_layers):
            gene = {
                'type': random.choice(self.block_types),
                'size': random.choice(self.layer_sizes)
            }
            genome.append(gene)
        
        return genome
    
    def _mutate_genome(self, genome):
        """Mutate genome with random changes"""
        mutated = genome.copy()
        
        # Mutation types
        mutation_type = random.choice(['change_type', 'change_size', 'add_layer', 'remove_layer'])
        
        if mutation_type == 'change_type' and len(mutated) > 0:
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx]['type'] = random.choice(self.block_types)
        
        elif mutation_type == 'change_size' and len(mutated) > 0:
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx]['size'] = random.choice(self.layer_sizes)
        
        elif mutation_type == 'add_layer' and len(mutated) < self.max_layers:
            new_gene = {
                'type': random.choice(self.block_types),
                'size': random.choice(self.layer_sizes)
            }
            idx = random.randint(0, len(mutated))
            mutated.insert(idx, new_gene)
        
        elif mutation_type == 'remove_layer' and len(mutated) > 2:
            idx = random.randint(0, len(mutated) - 1)
            mutated.pop(idx)
        
        return mutated
    
    def _crossover(self, parent1, parent2):
        """Crossover two genomes"""
        min_len = min(len(parent1), len(parent2))
        crossover_point = random.randint(1, min_len - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def evaluate_fitness(self, genome, train_data, val_data):
        """Evaluate architecture fitness"""
        try:
            # Create model
            model = EvolutionaryArchitecture(self.input_size, self.output_size, genome)
            
            # Quick training
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            # Train for few epochs
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                
                # Fitness = 1 / (1 + validation_loss)
                fitness = 1.0 / (1.0 + val_loss)
                
                # Penalty for complexity
                complexity_penalty = len(genome) * 0.01
                fitness -= complexity_penalty
                
                return max(fitness, 0.001)
        
        except Exception as e:
            print(f"Error evaluating genome: {e}")
            return 0.001
    
    def evolve_generation(self, train_data, val_data):
        """Evolve one generation"""
        print(f"🧬 Generation {self.generation}")
        
        # Evaluate fitness
        self.fitness_scores = []
        for i, genome in enumerate(self.population):
            fitness = self.evaluate_fitness(genome, train_data, val_data)
            self.fitness_scores.append(fitness)
            print(f"  Individual {i}: Fitness = {fitness:.4f}, Layers = {len(genome)}")
        
        # Selection (tournament selection)
        new_population = []
        
        # Keep best individuals (elitism)
        elite_count = self.population_size // 4
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < 0.7:  # Crossover probability
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < 0.3:  # Mutation probability
                child1 = self._mutate_genome(child1)
            if random.random() < 0.3:
                child2 = self._mutate_genome(child2)
            
            new_population.extend([child1, child2])
        
        # Update population
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Return best individual
        best_idx = np.argmax(self.fitness_scores)
        best_genome = self.population[best_idx]
        best_fitness = self.fitness_scores[best_idx]
        
        print(f"  🏆 Best fitness: {best_fitness:.4f}")
        return best_genome, best_fitness
    
    def _tournament_selection(self, tournament_size=3):
        """Tournament selection"""
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]


class TradingDataProcessor:
    """Process trading data for neural architecture search"""
    
    def __init__(self, data_path="eth_data.csv"):
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        
    def prepare_data(self, lookback=50, target_steps=5):
        """Prepare data for training"""
        # Calculate technical indicators
        self.data['sma_10'] = self.data['close'].rolling(10).mean()
        self.data['sma_20'] = self.data['close'].rolling(20).mean()
        self.data['rsi'] = self._calculate_rsi(self.data['close'])
        self.data['macd'] = self._calculate_macd(self.data['close'])
        
        # Create features
        features = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi', 'macd']
        
        X, y = [], []
        
        for i in range(lookback, len(self.data) - target_steps):
            # Features: lookback window
            feature_window = self.data[features].iloc[i-lookback:i].values.flatten()
            
            # Target: future price change
            current_price = self.data['close'].iloc[i]
            future_price = self.data['close'].iloc[i + target_steps]
            target = (future_price - current_price) / current_price
            
            X.append(feature_window)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Remove NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).flatten())
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        return (X_train, y_train), (X_test, y_test)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow


def main():
    """Main neural architecture search"""
    print("🧠 Neural Architecture Search V2.0")
    print("=" * 50)
    
    # Prepare data
    print("📊 Preparing data...")
    processor = TradingDataProcessor()
    train_data, val_data = processor.prepare_data()
    
    input_size = train_data[0].shape[1]
    output_size = 1
    
    print(f"Input size: {input_size}")
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    
    # Initialize evolution
    evolution = ArchitectureEvolution(input_size, output_size, population_size=15)
    
    # Evolve architectures
    best_architectures = []
    
    for generation in range(10):  # 10 generations
        best_genome, best_fitness = evolution.evolve_generation(train_data, val_data)
        
        best_architectures.append({
            'generation': generation,
            'genome': best_genome,
            'fitness': best_fitness,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Generation {generation} complete!\n")
    
    # Save results
    with open('neural_evolution_results.json', 'w') as f:
        json.dump(best_architectures, f, indent=2)
    
    # Train final best model
    print("🏆 Training final best architecture...")
    final_best = max(best_architectures, key=lambda x: x['fitness'])
    
    final_model = EvolutionaryArchitecture(input_size, output_size, final_best['genome'])
    
    # Extended training
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    final_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = final_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            final_model.eval()
            with torch.no_grad():
                val_outputs = final_model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss:.6f}")
            final_model.train()
    
    # Save final model
    torch.save(final_model.state_dict(), 'models/neural_architect_best.pth')
    
    print("\n✅ Neural Architecture Search Complete!")
    print(f"Best architecture fitness: {final_best['fitness']:.4f}")
    print(f"Best architecture layers: {len(final_best['genome'])}")
    
    # Print best architecture
    print("\n🏗️ Best Architecture:")
    for i, layer in enumerate(final_best['genome']):
        print(f"  Layer {i+1}: {layer['type']} ({layer['size']} units)")


if __name__ == "__main__":
    main()
