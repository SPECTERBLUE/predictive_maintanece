import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Function to simulate motor data with outliers, drift, and failure regions
def simulate_motor_data(
    n_samples=3600, 
    failures=[800, 1900, 3100], 
    pre_fail_window=30, 
    post_fail_window=20,
    outlier_prob=0.02,     # 2% of readings are outliers
    drift_strength=0.0005  # gradual drift over time
):
    np.random.seed(42)
    time = np.arange(n_samples)
    temperature = 60 + np.random.normal(0, 1.5, n_samples)
    vibration = 0.12 + np.random.normal(0, 0.02, n_samples)
    labels = np.zeros(n_samples)  # 0=normal,1=pre-failure,2=post-failure

    # Add gradual drift to simulate sensor bias
    drift = np.linspace(0, n_samples*drift_strength, n_samples)
    temperature += drift

    # Inject failure patterns
    for f in failures:
        start_pre = max(f - pre_fail_window, 0)
        end_post = min(f + post_fail_window, n_samples)

        # Pre-failure: gradual rise
        temperature[start_pre:f] += np.linspace(0, 10, f - start_pre)
        vibration[start_pre:f] += np.linspace(0, 0.15, f - start_pre) + np.random.normal(0, 0.01, f - start_pre)
        labels[start_pre:f] = 1

        # Failure: sharp spike
        temperature[f] += 15
        vibration[f] += 0.3
        labels[f] = 1

        # Post-failure: unstable
        temp_decay = np.linspace(10, -5, end_post - f)
        vibration_noise = np.random.normal(0.2, 0.1, end_post - f)
        temperature[f:end_post] += temp_decay
        vibration[f:end_post] += vibration_noise
        labels[f:end_post] = 2

    # Add random outliers (simulate electrical spikes, etc.)
    n_outliers = int(outlier_prob * n_samples)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)

    for idx in outlier_indices:
        if np.random.rand() < 0.5:
            temperature[idx] += np.random.choice([-1, 1]) * np.random.uniform(10, 25)
        else:
            vibration[idx] += np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0)

    # Combine into dataframe
    df = pd.DataFrame({
        'time_min': time * 2,
        'temperature': temperature,
        'vibration': vibration,
        'label': labels.astype(int)
    })
    return df

# Generate multiple motors with different behavior
motors = []
for motor_id in range(1, 6):
    df_motor = simulate_motor_data(
        n_samples=3600, 
        failures=np.random.choice(range(500,3500,500), 3, replace=False),
        outlier_prob=np.random.uniform(0.01, 0.05)
    )
    df_motor['motor_id'] = motor_id
    motors.append(df_motor)

# Combine into one dataset
full_df = pd.concat(motors).reset_index(drop=True)

# Save to CSV
full_df.to_csv("multi_motor_data.csv", index=False)

# Visualization: one sample motor
df_example = full_df[full_df['motor_id'] == 1]

plt.figure(figsize=(12,6))
plt.plot(df_example['time_min'], df_example['temperature'], label='Temperature (°C)')
plt.plot(df_example['time_min'], df_example['vibration']*200, label='Vibration (scaled)')
plt.scatter(df_example.loc[df_example['label']==1, 'time_min'], 
            df_example.loc[df_example['label']==1, 'temperature'], color='orange', s=10, label='Pre-failure')
plt.scatter(df_example.loc[df_example['label']==2, 'time_min'], 
            df_example.loc[df_example['label']==2, 'temperature'], color='red', s=10, label='Post-failure')
plt.xlabel('Time (minutes)')
plt.legend()
plt.title('Simulated Motor Sensor Data with Random Outliers (Motor 1)')
plt.show()

from tabulate import tabulate

print("Simulated Multi-Motor Dataset:")
print(tabulate(full_df, headers='keys', tablefmt='psql'))

#---------------------------------------------------------------------------------------
# Load the CSV
full_df = pd.read_csv("multi_motor_data.csv")

# Unique motors
motor_ids = full_df['motor_id'].unique()
num_motors = len(motor_ids)

plt.figure(figsize=(75, 30))

# Get colormap and sample colors for each motor
cmap = matplotlib.colormaps['tab10']  # just get the colormap
colors = [cmap(i / max(1, num_motors-1)) for i in range(num_motors)]  # scale 0-1

# Plot Temperature and Vibration for each motor
for i, motor_id in enumerate(motor_ids):
    motor_data = full_df[full_df['motor_id'] == motor_id]
    
    # Temperature
    plt.plot(
        motor_data['time_min'], 
        motor_data['temperature'], 
        label=f'Motor {motor_id} Temp', 
        color=colors[i],
        linestyle='-',
        marker='o'
    )
    
    # Vibration
    plt.plot(
        motor_data['time_min'], 
        motor_data['vibration'], 
        label=f'Motor {motor_id} Vib', 
        color=colors[i], 
        marker='x', 
        linestyle='--'
    )

plt.title("Multi-Motor Sensor Data Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature / Vibration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("multi_motor_plot.png")
print("Plot saved as multi_motor_plot.png")

