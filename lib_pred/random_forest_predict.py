# random_forest_predict.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_random_forest(input_csv="normalized_motor_data.csv"):
    # Load and sort data
    df = pd.read_csv(input_csv)
    df = df.sort_values(['motor_id', 'time_min']).reset_index(drop=True)

    # Use current readings to predict next readings
    df['temp_next'] = df.groupby('motor_id')['temperature'].shift(-1)
    df['vib_next'] = df.groupby('motor_id')['vibration'].shift(-1)
    df.dropna(inplace=True)

    X = df[['temperature', 'vibration']]
    y_temp = df['temp_next']
    y_vib = df['vib_next']

    # Split into train/test
    X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    _, _, y_train_vib, y_test_vib = train_test_split(X, y_vib, test_size=0.2, random_state=42)

    # Train separate models for temperature and vibration
    model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_vib = RandomForestRegressor(n_estimators=100, random_state=42)

    model_temp.fit(X_train, y_train_temp)
    model_vib.fit(X_train, y_train_vib)

    # Make predictions
    preds_temp = model_temp.predict(X_test)
    preds_vib = model_vib.predict(X_test)

    # Evaluate performance
    mse_temp = mean_squared_error(y_test_temp, preds_temp)
    rmse_temp = np.sqrt(mse_temp)
    mae_temp = mean_absolute_error(y_test_temp, preds_temp)
    r2_temp = r2_score(y_test_temp, preds_temp)

    mse_vib = mean_squared_error(y_test_vib, preds_vib)
    rmse_vib = np.sqrt(mse_vib)
    mae_vib = mean_absolute_error(y_test_vib, preds_vib)
    r2_vib = r2_score(y_test_vib, preds_vib)

    # Print results
    print(f"--- Temperature Prediction ---")
    print(f"MSE  : {mse_temp:.6f}")
    print(f"RMSE : {rmse_temp:.6f}")
    print(f"MAE  : {mae_temp:.6f}")
    print(f"R²   : {r2_temp:.6f}")

    print(f"\n--- Vibration Prediction ---")
    print(f"MSE  : {mse_vib:.6f}")
    print(f"RMSE : {rmse_vib:.6f}")
    print(f"MAE  : {mae_vib:.6f}")
    print(f"R²   : {r2_vib:.6f}")

    # -------------------------------
    # Visualization: Actual vs Predicted
    # -------------------------------
    plt.figure(figsize=(12, 6))

    # Temperature subplot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_temp, preds_temp, alpha=0.6, color='red')
    plt.plot([y_test_temp.min(), y_test_temp.max()],
             [y_test_temp.min(), y_test_temp.max()],
             'k--', lw=2)
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title("Temperature: Actual vs Predicted")
    plt.grid(True)

    # Vibration subplot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_vib, preds_vib, alpha=0.6, color='blue')
    plt.plot([y_test_vib.min(), y_test_vib.max()],
             [y_test_vib.min(), y_test_vib.max()],
             'k--', lw=2)
    plt.xlabel("Actual Vibration")
    plt.ylabel("Predicted Vibration")
    plt.title("Vibration: Actual vs Predicted")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("random_forest_plot.png")
    print("Plot saved as multi_motor_plot.png")

if __name__ == "__main__":
    train_random_forest()
