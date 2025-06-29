import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def predict_tomorrow_price():
    
    ticker = "COCHINSHIP.NS"
    try:
        data = yf.download(ticker, period="3y", interval="1d")
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance")
        print(f"Successfully downloaded {len(data)} days of data")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    
    def add_features(df):
        df = df.copy()
        # Moving Averages
        df['MA5'] = df['Close'].rolling(5, min_periods=3).mean()
        df['MA20'] = df['Close'].rolling(20, min_periods=10).mean()
        # Momentum
        df['Momentum'] = df['Close'].pct_change(5)
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(5).std()
        # Daily Range
        df['Range'] = (df['High'] - df['Low']) / df['Close']
        return df.dropna()

    data = add_features(data)
    if len(data) < 50:
        print("Insufficient data after feature engineering")
        return None

    # Target = Next day's closing price
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()

    features = ['MA5', 'MA20', 'Momentum', 'Volatility', 'Range', 'Volume']
    X = data[features]
    y = data['Target']

    
    split = int(len(data)*0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train Model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_preds)
    print(f"\nModel Evaluation:")
    print(f"Mean Absolute Error: ₹{mae:.2f}")
    print(f"Average Price: ₹{y_test.mean():.2f}")
    print(f"Error Percentage: {100*mae/y_test.mean():.1f}%")

    # Predict Tomorrow
    latest = X.iloc[[-1]]
    tomorrow_pred = float(model.predict(latest)[0])  
    last_date = data.index[-1].strftime('%d %b %Y')
    last_close = float(data['Close'].iloc[-1])  
    change_pct = 100*(tomorrow_pred/last_close-1)

    print(f"\nPrediction Results:")
    print(f"Today's Close ({last_date}): ₹{last_close:.2f}")
    print(f"Predicted Tomorrow's Close: ₹{tomorrow_pred:.2f}")
    print(f"Expected Change: {change_pct:.2f}%")

    plt.figure(figsize=(12,6))
    plt.plot(data.index[-30:], data['Close'][-30:], 'b-', label='Actual Price')
    plt.plot([data.index[-1], data.index[-1] + pd.Timedelta(days=1)], 
             [last_close, tomorrow_pred], 'r--', label='Prediction')
    plt.scatter(data.index[-1] + pd.Timedelta(days=1), tomorrow_pred, color='red', s=100)
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.legend()
    plt.grid()
    plt.show()

    return {
        'last_close': last_close,
        'predicted_price': tomorrow_pred,
        'change_pct': change_pct
    }

prediction = predict_tomorrow_price()
