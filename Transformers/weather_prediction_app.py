import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Preprocess data
def preprocess_data(df, feature, seq_len=10):
    df = df[[feature]].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i:i+seq_len])
        y.append(scaled_data[i+seq_len])
    
    return np.array(X), np.array(y), scaler

# Build models
def create_model(model_type, input_shape):
    model = Sequential([
        (SimpleRNN if model_type == "RNN" else LSTM)(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        (SimpleRNN if model_type == "RNN" else LSTM)(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Streamlit UI
st.title("📊 Weather Forecasting with RNN & LSTM")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        feature = st.selectbox("🔢 Select Feature for Prediction", numeric_cols)
        model_type = st.radio("🛠 Choose Model", ["RNN", "LSTM"])

        if st.button("🚀 Train Model"):
            X, y, scaler = preprocess_data(df, feature)
            split_idx = int(0.8 * len(X))
            X_train, y_train, X_test, y_test = X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

            model = create_model(model_type, (X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

            predictions = scaler.inverse_transform(model.predict(X_test))
            actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Calculate Metrics
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predictions)

            # Display Accuracy Metrics
            st.write("### 📊 Model Performance Metrics")
            st.write(f"🔹 **Mean Squared Error (MSE):** {mse:.4f}")
            st.write(f"🔹 **Root Mean Squared Error (RMSE):** {rmse:.4f}")
            st.write(f"🔹 **Mean Absolute Error (MAE):** {mae:.4f}")

            # Plot results
            fig, ax = plt.subplots()
            ax.plot(actual, label="Actual", linestyle="dotted", color="blue")
            ax.plot(predictions, label="Predicted", color="red")
            ax.legend()
            st.pyplot(fig)

            # Show sample predicted values
            results_df = pd.DataFrame({"Actual": actual.flatten(), "Predicted": predictions.flatten()})
            st.write("### 🔍 Sample Predictions")
            st.write(results_df.head(10))

            model.save("weather_model.h5")
            st.success(f"✅ {model_type} Model Trained & Saved!")
    else:
        st.error("No numerical columns found in the dataset. Please upload a valid file.")
