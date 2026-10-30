# Stock Price Prediction Model

## What This Project Does
This project uses **Machine Learning** to predict stock prices using Python and scikit-learn.

## Features
- Downloads real stock data from Yahoo Finance
- Trains a Random Forest model on historical prices
- Makes predictions for future stock prices
- Shows accuracy metrics (R² Score, RMSE, MAE)
- Fully version controlled on GitHub

## Installation

### Step 1: Install Python
Download from python.org and install

### Step 2: Install Required Libraries
```bash
pip install -r requirements.txt
```

## How to Use

### Step 1: Train the Model
```bash
python train_model.py
```

This downloads Apple stock data and trains the model. You'll see:
- Training progress
- Accuracy metrics
- Model saved confirmation

### Step 2: Make Predictions
```bash
python predict.py
```

This uses the trained model to predict tomorrow's stock price for Apple.

## Project Structure