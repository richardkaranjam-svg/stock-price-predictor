import joblib
import numpy as np
import yfinance as yf
import pandas as pd

# This file uses the trained model to make predictions

def get_latest_stock_data(ticker, days=30):
    """
    Get the latest stock data for prediction
    """
    data = yf.download(ticker, period='30d', progress=False)
    return data


def predict_next_price(ticker='AAPL'):
    """
    Predict tomorrow's stock price
    """
    
    print(f"\n{'='*50}")
    print(f"Predicting {ticker} Stock Price")
    print(f"{'='*50}")
    
    # Load the trained model
    try:
        model = joblib.load('stock_price_model.pkl')
        print("✓ Model loaded!")
    except:
        print("❌ Model not found! Train the model first by running: python train_model.py")
        return
    
    # Get latest data
    print(f"Downloading latest {ticker} data...")
    data = get_latest_stock_data(ticker)
    
    # Create features (same as training)
    features = pd.DataFrame()
    features['Close'] = data['Close']
    features['Volume'] = data['Volume']
    features['High'] = data['High']
    features['Low'] = data['Low']
    features['MA_5'] = data['Close'].rolling(window=5).mean()
    features['MA_20'] = data['Close'].rolling(window=20).mean()
    
    # Get the most recent data
    latest_features = features.iloc[-1:].dropna()
    
    # Make prediction
    predicted_price = model.predict(latest_features)[0]
    current_price = data['Close'].iloc[-1]
    price_change = predicted_price - current_price
    change_percent = (price_change / current_price) * 100
    
    print(f"\nCurrent Price: ${current_price:.2f}")
    print(f"Predicted Tomorrow's Price: ${predicted_price:.2f}")
    print(f"Expected Change: ${price_change:.2f} ({change_percent:+.2f}%)")
    
    if price_change > 0:
        print("📈 Stock expected to GO UP")
    else:
        print("📉 Stock expected to GO DOWN")
    
    print(f"{'='*50}\n")
    
    return predicted_price


if __name__ == "__main__":
    # Make a prediction for Apple stock
    predict_next_price('AAPL')