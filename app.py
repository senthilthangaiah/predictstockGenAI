import streamlit as st
import torch
import numpy as np
import pandas as pd
from torch import nn

# Transformer Model definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, forecast_steps):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(model_dim, output_dim * forecast_steps)  # Output for each forecast step

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.model_dim)
        src = src + self.pos_encoder
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output.view(output.size(0), -1, output.size(-1))  # Reshape to separate each forecast step

# Load the model
model = TransformerModel(input_dim=5, model_dim=64, num_heads=4, num_layers=2, output_dim=1, forecast_steps=5)
model.load_state_dict(torch.load('transformer_stock_model.pth'))
model.eval()

# Streamlit app
st.title("Stock Price Prediction for Next 5 Days with Transformer")
st.write("Enter the stock symbol to predict the next 5 closing prices.")

# Input stock symbol
stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL)", "AAPL")

if st.button("Predict"):
    try:
        # Fetch historical data for the given stock symbol
        # Here you can use your preferred method to fetch data from an API or a dataset
        # For simplicity, we assume data is fetched and preprocessed here
        # Example: df = fetch_stock_data(stock_symbol)
        # Example: last_seq = preprocess_data(df)
        # We'll use random data for demonstration purposes
        last_seq = np.random.rand(50, 5)  # Random data with 50 time steps and 5 features
        input_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            predictions = model(input_tensor)
        
        st.write("Predicted closing prices for the next 5 days:")
        for i, prediction in enumerate(predictions.squeeze()):
            st.write(f"Day {i+1}: {prediction.tolist()}")
    except Exception as e:
        st.write(f"Error: {e}")

