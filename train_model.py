from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pandas as pd
from data_loader import download_stock_data, prepare_data, create_features

# This file trains the machine learning model

def train_stock_model(ticker='AAPL', start_date='2020-01-01', end_date='2023-01-01'):
    """
    Download data and train a stock price prediction model
    """
    
    print("="*50)
    print(f"Training Stock Price Model for {ticker}")
    print("="*50)
    
    # Step 1: Download stock data
    data = download_stock_data(ticker, start_date, end_date)
    
    # Step 2: Prepare the data
    data = prepare_data(data)
    
    # Step 3: Create features (inputs)
    features = create_features(data)
    
    # Step 4: Get target (what we're predicting - tomorrow's price)
    target = data['Price_Change'][features.index]
    
    # Step 5: Split into training (70%) and testing (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        target, 
        test_size=0.2,  # 20% for testing
        random_state=42
    )
    
    print(f"\nTraining data: {len(X_train)} days")
    print(f"Testing data: {len(X_test)} days")
    
    # Step 6: Create the model
    model = RandomForestRegressor(
        n_estimators=100,      # Use 100 trees
        random_state=42,
        max_depth=20
    )
    
    # Step 7: Train the model
    print("\nTraining model (this takes a few seconds)...")
    model.fit(X_train, y_train)
    print("✓ Model trained!")
    
    # Step 8: Test the model
    y_pred = model.predict(X_test)
    
    # Step 9: Calculate accuracy
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL RESULTS")
    print("="*50)
    print(f"R² Score: {r2:.2%}")           # How accurate (0-100%)
    print(f"RMSE: ${rmse:.2f}")            # Error in dollars
    print(f"MAE: ${mae:.2f}")              # Average error
    print("="*50)
    
    # Step 10: Save the model
    joblib.dump(model, 'stock_price_model.pkl')
    print("\n✓ Model saved as 'stock_price_model.pkl'")
    
    return model, features.columns


if __name__ == "__main__":
    # Run the training
    model, features = train_stock_model()