# random_forest_predict.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

def train_random_forest_classification(input_csv="multi_motor_data.csv"):
    # Load data
    df = pd.read_csv(input_csv)

    # Features and target
    features = ['temperature', 'vibration']
    target = 'label'

    X = df[features]
    y = df[target]

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # -------------------------------
    # Evaluate Performance
    # -------------------------------
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print("=== Random Forest Classification Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)

    # -------------------------------
    # Visualization: Confusion Matrix
    # -------------------------------
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pre-failure', 'Post-failure'],
                yticklabels=['Normal', 'Pre-failure', 'Post-failure'])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Random Forest Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("multi_motor_classification_plot.png")
    print("Plot saved as multi_motor_classification_plot.png")

    # -------------------------------
    # Plot 2: Actual vs Predicted Labels (Timeline)
    # -------------------------------
    # Sort by index to simulate time sequence for visualization
    y_test_reset = y_test.reset_index(drop=True)
    y_pred_reset = pd.Series(y_pred, name="Predicted").reset_index(drop=True)

    plt.figure(figsize=(10,5))
    plt.plot(y_test_reset, label='Actual Label', linestyle='-', marker='o', alpha=0.7)
    plt.plot(y_pred_reset, label='Predicted Label', linestyle='--', marker='x', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Motor Status (0=Normal, 1=Pre-failure, 2=Post-failure)")
    plt.title("Actual vs Predicted Motor Status")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("multi_motor_actual_vs_predicted.png")
    print("Actual vs Predicted plot saved as multi_motor_actual_vs_predicted.png")

if __name__ == "__main__":
    train_random_forest_classification()
