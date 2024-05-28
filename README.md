# Stock Price Prediction using Transformer-based Generative AI

This repository contains code for predicting stock prices for the next 5 days using a Transformer-based generative AI model. The model is trained on historical stock data and deployed using Streamlit for easy inference.

## Overview

The project consists of two main components:

1. **Training Script (train.py)**:
   - Loads and preprocesses historical stock data.
   - Implements a Transformer model architecture for stock price prediction.
   - Trains the model using the provided data.
   - Saves the trained model for later use.

2. **Inference Script (app.py)**:
   - Loads the trained Transformer model.
   - Implements a Streamlit web application for user interaction.
   - Takes user input for a stock symbol and predicts the next 5 closing prices.
   - Displays the predictions to the user.

## Usage

1. **Training the Model**:
   - Run `train.py` to train the model on your historical stock data.
   - Adjust hyperparameters as needed in the script.
   - The trained model will be saved as `transformer_stock_model.pth`.

2. **Running the Streamlit App**:
   - Ensure you have all dependencies installed (`torch`, `numpy`, `pandas`, `streamlit`).
   - Run `streamlit run app.py` to start the Streamlit web application.
   - Enter a stock symbol and click "Predict" to see the predicted closing prices for the next 5 days.

## File Structure

