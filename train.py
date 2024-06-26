import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Load and preprocess the data
def load_and_preprocess_data(file_path, seq_len=50):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['symbol', 'date'], inplace=True)
    
    grouped = df.groupby('symbol')
    X, y = [], []

    for _, group in grouped:
        data = group[['open', 'close', 'low', 'high', 'volume']].values
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len, 1])  # Predict the 'close' price

    return np.array(X), np.array(y)

# Custom Dataset class
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.model_dim)
        src = src + self.pos_encoder
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

# Hyperparameters
input_dim = 5  # open, close, low, high, volume
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 1
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Prepare data
X, y = load_and_preprocess_data('price.csv')
dataset = StockDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')

# Save the model
torch.save(model.state_dict(), 'transformer_stock_model.pth')
print("Model saved successfully.")
