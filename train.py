import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pickle
import time

# Set random seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load the data
data = pd.read_csv('train_data.csv')
data.columns = data.columns.str.strip()

# Convert the Date column to datetime
data['Date Logs'] = pd.to_datetime(data['Date Logs'], format='%m/%d/%Y')

# Check for missing values
if data.isnull().values.any():
    data = data.dropna()

# Sort the data by date
data = data.sort_values('Date Logs')

data = pd.get_dummies(
    data, columns=['Month', 'Weekday or Weekend', 'Type of Day'])
data.columns = data.columns.str.strip()

# Select relevant features
features = ['Month_April', 'Month_August', 'Month_December', 'Month_February', 
            'Month_January', 'Month_July', 'Month_June', 'Month_March', 
            'Month_May', 'Month_November', 'Month_October', 'Month_September',
            'Weekday or Weekend_Weekday', 'Weekday or Weekend_Weekend', 
            'Type of Day_Normal Day', 'Type of Day_Regular Holiday', 
            'Type of Day_Special Non-working Holiday',
            'Mean Temperature (Degree Celsius)', 'Rainfall(mm)', 
            'Relative Humidity (%)', 'Windspeed (m/s)']
target = 'DemandLoad'

# Prepare input and output data
X = data[features].values
y = data[target].values

# Normalize the features and the target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Create sequences
def create_sequences(X, y, time_steps=7):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 7
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Use all data for training
X_train, y_train = X_seq, y_seq

# Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(
        X_train.shape[1], X_train.shape[2])),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mape')

# Record the start time
start_time = time.time()

# Train the model with callback to record loss over epochs
history = model.fit(X_train, y_train, epochs=70, batch_size=32, verbose=1)

# Record the end time
end_time = time.time()

# Calculate the total training time
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Save the model
model.save("dbn_model.h5")

# Save the scalers
with open('scaler_dbn_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_dbn_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# Save training loss to a CSV file for visualization
loss_data = pd.DataFrame({
    "Epoch": range(1, len(history.history['loss']) + 1),
    "Loss": history.history['loss']
})
loss_data.to_csv('training_loss.xlsx', index=False)

# Output CSV path
print("Training loss saved to 'training_loss.xlsx'")
