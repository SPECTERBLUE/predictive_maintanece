
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def train_lstm_predictor(input_csv="multi_motor_data.csv", sequence_length=20):
    # -------------------------------
    # Load and preprocess data
    # -------------------------------
    df = pd.read_csv(input_csv)

    # Use only relevant features
    features = ['temperature', 'vibration']
    target = 'label'

    X = df[features].values
    y = df[target].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------
    # Prepare sequences for LSTM
    # -------------------------------
    def create_sequences(data, labels, seq_length):
        Xs, ys = [], []
        for i in range(len(data) - seq_length):
            Xs.append(data[i:(i + seq_length)])
            ys.append(labels[i + seq_length])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

    # Train-test split (80/20)
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # -------------------------------
    # Build LSTM Model
    # -------------------------------
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, len(features)), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: normal, pre-failure, post-failure
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------------
    # Train the model
    # -------------------------------
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # -------------------------------
    # Evaluate model
    # -------------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ LSTM Test Accuracy: {test_acc:.4f}")

    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Pre-failure', 'Post-failure'],
        yticklabels=['Normal', 'Pre-failure', 'Post-failure']
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("LSTM Model - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("lstm_confusion_matrix.png")
    print("Confusion matrix plot saved as lstm_confusion_matrix.png")

    # -------------------------------
    # Actual vs Predicted Timeline Plot
    # -------------------------------
    y_test_reset = pd.Series(y_test, name="Actual").reset_index(drop=True)
    y_pred_reset = pd.Series(y_pred, name="Predicted").reset_index(drop=True)

    plt.figure(figsize=(10,5))
    plt.plot(y_test_reset, label='Actual Label', linestyle='-', marker='o', alpha=0.7)
    plt.plot(y_pred_reset, label='Predicted Label', linestyle='--', marker='x', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Motor Status (0=Normal, 1=Pre-failure, 2=Post-failure)")
    plt.title("LSTM Model - Actual vs Predicted Motor Status")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lstm_actual_vs_predicted.png")
    print("Actual vs Predicted plot saved as lstm_actual_vs_predicted.png")

if __name__ == "__main__":
    train_lstm_predictor()
