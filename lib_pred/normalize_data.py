import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(input_csv="multi_motor_data.csv", output_csv="normalized_motor_data.csv"):
    df = pd.read_csv(input_csv)

    # Select features to normalize
    features = ['temperature', 'vibration']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Save normalized dataset
    df.to_csv(output_csv, index=False)
    print(f"✅ Normalized data saved to {output_csv}")

if __name__ == "__main__":
    normalize_data()
