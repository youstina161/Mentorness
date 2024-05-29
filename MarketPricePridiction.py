import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import matplotlib.pyplot as plt

# 1. Data Preprocessing
def preprocess_data(df):
    # Handling missing values
    df.fillna(method='ffill', inplace=True)
    
    # Encoding categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders

# 2. Exploratory Data Analysis (EDA)
def plot_eda(df, column):
    df[column].plot(figsize=(12, 6))
    plt.title(f'Time Series Plot for {column}')
    plt.show()

# 3. Feature Engineering
def create_features(df, column):
    df['lag1'] = df[column].shift(1)
    df['rolling_mean_3'] = df[column].rolling(window=3).mean()
    df['rolling_std_3'] = df[column].rolling(window=3).std()
    df['month'] = df.index.month
    df.dropna(inplace=True)
    return df

# Load the dataset
df = pd.read_csv('\\Users\\youst\Desktop\MarkretPicePrediction\MarketPricePrediction.csv', index_col='date', parse_dates=True)
df, label_encoders = preprocess_data(df)

# EDA
plot_eda(df, 'market_quantity')
plot_eda(df, 'market_price')

# Feature Engineering
df = create_features(df, 'market_quantity')

# Train-test split
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# 4. Model Selection and Training
# ARIMA Model
def train_arima(train):
    arima_model = ARIMA(train['market_quantity'], order=(5, 1, 0)).fit()
    return arima_model

# Prophet Model
def train_prophet(train):
    prophet_data = train.reset_index().rename(columns={'date': 'ds', 'market_quantity': 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    return prophet_model

# LSTM Model
def train_lstm(train):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train['market_quantity'].values.reshape(-1, 1))
    
    X_train = []
    y_train = []
    for i in range(1, len(train_scaled)):
        X_train.append(train_scaled[i-1:i])
        y_train.append(train_scaled[i])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=2)
    
    return lstm_model, scaler

# Train models
arima_model = train_arima(train)
prophet_model = train_prophet(train)
lstm_model, scaler = train_lstm(train)

# 5. Model Evaluation
def evaluate_model(model, test, model_type='arima'):
    if model_type == 'arima':
        predictions = model.forecast(steps=len(test))
    elif model_type == 'prophet':
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        predictions = forecast['yhat'].tail(len(test)).values
    elif model_type == 'lstm':
        test_scaled = scaler.transform(test['market_quantity'].values.reshape(-1, 1))
        X_test = []
        for i in range(1, len(test_scaled)):
            X_test.append(test_scaled[i-1:i])
        X_test = np.array(X_test)
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
    return predictions

# Evaluate ARIMA
arima_preds = evaluate_model(arima_model, test, model_type='arima')
mae_arima = mean_absolute_error(test['market_quantity'], arima_preds)
mse_arima = mean_squared_error(test['market_quantity'], arima_preds)
rmse_arima = np.sqrt(mse_arima)

# Evaluate Prophet
prophet_preds = evaluate_model(prophet_model, test, model_type='prophet')
mae_prophet = mean_absolute_error(test['market_quantity'], prophet_preds)
mse_prophet = mean_squared_error(test['market_quantity'], prophet_preds)
rmse_prophet = np.sqrt(mse_prophet)

# Evaluate LSTM
lstm_preds = evaluate_model(lstm_model, test, model_type='lstm')
mae_lstm = mean_absolute_error(test['market_quantity'][1:], lstm_preds)
mse_lstm = mean_squared_error(test['market_quantity'][1:], lstm_preds)
rmse_lstm = np.sqrt(mse_lstm)

print(f'ARIMA MAE: {mae_arima}, MSE: {mse_arima}, RMSE: {rmse_arima}')
print(f'Prophet MAE: {mae_prophet}, MSE: {mse_prophet}, RMSE: {rmse_prophet}')
print(f'LSTM MAE: {mae_lstm}, MSE: {mse_lstm}, RMSE: {rmse_lstm}')

# Save the best model (assuming ARIMA in this example)
best_model = arima_model
joblib.dump(best_model, 'best_model.pkl')

# 6. Fine-tuning and Validation
# Assuming we have a validation set for final evaluation
validation = pd.read_csv('your_validation_data.csv', index_col='date', parse_dates=True)
validation, _ = preprocess_data(validation)

# Evaluate the best model on the validation set
validation_preds = evaluate_model(best_model, validation, model_type='arima')
mae_validation = mean_absolute_error(validation['market_quantity'], validation_preds)
mse_validation = mean_squared_error(validation['market_quantity'], validation_preds)
rmse_validation = np.sqrt(mse_validation)

print(f'Validation MAE: {mae_validation}, MSE: {mse_validation}, RMSE: {rmse_validation}')
