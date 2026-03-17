import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def predict_symbol(symbol):
    print(f"Fetching data for {symbol}...")
    df = yf.Ticker(symbol).history(period="60d", interval="1h")

    if df.empty:
        return "Error: Symbol not found or no data."

    # --- 1. Data Processing ---
    # Resample 1-hour data to 4-hour candles (H4)
    df = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    df.columns = df.columns.str.lower()
    df = df.dropna()

    # --- 2. Feature Engineering ---
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.ema(length=50, append=True, col_names='EMA_50')

    for i in range(1, 6):
        df[f'price_change_lag_{i}'] = df['close'].diff(i)

    df.dropna(inplace=True)

    # --- 3. Target Definition & Data Splitting ---
    prediction_day_date = df.index.max().normalize()
    historical_df = df[df.index < prediction_day_date].copy()
    prediction_df = df[df.index.normalize() == prediction_day_date].copy()

    if prediction_df.empty:
        return "Error: No data available for prediction day."

    daily_close_map = historical_df.resample('D')['close'].last()
    historical_df['target_eod_close'] = historical_df.index.normalize().map(daily_close_map)
    historical_df.dropna(inplace=True)

    # --- 4. Data Preparation for RNN Model ---
    features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 'target_eod_close']
    features = [col for col in historical_df.columns if col not in features_to_exclude]
    X = historical_df[features]
    y = historical_df[['target_eod_close']]

    train_size = int(len(X) * 0.8)
    X_train_df, X_test_df = X[:train_size], X[train_size:]
    y_train_df, y_test_df = y[:train_size], y[train_size:]

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_df)
    X_test_scaled = x_scaler.transform(X_test_df)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_df)
    y_test_scaled = y_scaler.transform(y_test_df)

    sequence_length = 10

    def create_sequences(X_data, y_data, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X_data) - seq_length):
            X_seq.append(X_data[i:(i + seq_length)])
            y_seq.append(y_data[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

    # --- 5. Model Training ---
    model = Sequential([
        SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        SimpleRNN(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    # Reduced verbosity for web app
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # --- 6. Prediction ---
    full_features_df = df.drop(columns=features_to_exclude, errors='ignore')
    latest_sequence_df = full_features_df.iloc[-sequence_length:]
    latest_sequence_scaled = x_scaler.transform(latest_sequence_df)
    input_for_prediction = np.expand_dims(latest_sequence_scaled, axis=0)
    
    todays_eod_prediction_scaled = model.predict(input_for_prediction)
    todays_eod_prediction = y_scaler.inverse_transform(todays_eod_prediction_scaled)

    return f"{todays_eod_prediction[0][0]:.4f}"
