#!/usr/bin/env python
import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    print(f"Downloaded {len(data)} days of data")
    return data


def prepare_data(data):
    """Prepare data for the machine learning model"""
    df = data.copy()
    df['Price_Change'] = df['Close'].shift(-1)
    df = df[:-1]
    return df


def create_features(data):
    """Create features for the model"""
    features = pd.DataFrame()
    features['Close'] = data['Close']
    features['Volume'] = data['Volume']
    features['High'] = data['High']
    features['Low'] = data['Low']
    features['MA_5'] = data['Close'].rolling(window=5).mean()
    features['MA_20'] = data['Close'].rolling(window=20).mean()
    features = features.dropna()
    return features