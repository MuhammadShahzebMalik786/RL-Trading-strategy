import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.softmax(out)

class ANNPredictor(nn.Module):
    def __init__(self, input_size: int = 250, hidden_sizes: list = [128, 64, 32], output_size: int = 3):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CNNPredictor(nn.Module):
    def __init__(self, input_channels: int = 5, sequence_length: int = 50, output_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size
        conv_output_size = 128 * (sequence_length // 8)  # After 3 pooling operations
        
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return self.softmax(x)

class RNNPredictor(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return self.softmax(out)

class DeepLearningEnsemble:
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.models = {}
        self.trained = False
        
    def prepare_data(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training"""
        # Normalize data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data) - 1):
            X.append(scaled_data[i-self.sequence_length:i])
            
            # Create labels: 0=sell, 1=hold, 2=buy
            future_price = scaled_data[i+1, 3]  # Next close price
            current_price = scaled_data[i, 3]   # Current close price
            
            if future_price > current_price * 1.001:  # 0.1% threshold
                y.append(2)  # Buy
            elif future_price < current_price * 0.999:
                y.append(0)  # Sell
            else:
                y.append(1)  # Hold
        
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    def train_models(self, data: np.ndarray, epochs: int = 50):
        """Train all deep learning models"""
        X, y = self.prepare_data(data)
        
        # Initialize models
        self.models = {
            'lstm': LSTMPredictor(),
            'ann': ANNPredictor(input_size=self.sequence_length * 5),
            'cnn': CNNPredictor(sequence_length=self.sequence_length),
            'rnn': RNNPredictor()
        }
        
        criterion = nn.CrossEntropyLoss()
        
        for name, model in self.models.items():
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            print(f"Training {name.upper()}...")
            
            for epoch in range(epochs):
                if name == 'ann':
                    # Flatten input for ANN
                    X_flat = X.view(X.size(0), -1)
                    outputs = model(X_flat)
                elif name == 'cnn':
                    # Transpose for CNN (batch, channels, sequence)
                    X_cnn = X.transpose(1, 2)
                    outputs = model(X_cnn)
                else:
                    outputs = model(X)
                
                loss = criterion(outputs, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.trained = True
        print("All models trained successfully!")
    
    def predict(self, recent_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all models"""
        if not self.trained:
            raise ValueError("Models must be trained first!")
        
        # Prepare input
        scaled_data = self.scaler.transform(recent_data[-self.sequence_length:])
        X = torch.FloatTensor(scaled_data).unsqueeze(0)
        
        predictions = {}
        
        with torch.no_grad():
            for name, model in self.models.items():
                if name == 'ann':
                    X_flat = X.view(1, -1)
                    pred = model(X_flat)
                elif name == 'cnn':
                    X_cnn = X.transpose(1, 2)
                    pred = model(X_cnn)
                else:
                    pred = model(X)
                
                predictions[name] = pred.numpy()[0]
        
        return predictions
    
    def get_ensemble_signal(self, predictions: Dict[str, np.ndarray]) -> int:
        """Combine predictions from all models"""
        # Weighted voting (LSTM gets higher weight)
        weights = {'lstm': 0.4, 'ann': 0.2, 'cnn': 0.2, 'rnn': 0.2}
        
        ensemble_probs = np.zeros(3)
        for name, pred in predictions.items():
            ensemble_probs += pred * weights[name]
        
        return np.argmax(ensemble_probs)  # 0=sell, 1=hold, 2=buy
