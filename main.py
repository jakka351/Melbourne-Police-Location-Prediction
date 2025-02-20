import tkinter as tk
from tkinter import ttk, filedialog
import sys
import threading
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------
# 1. LOAD & PARSE CSV
# -------------------------------------------------
def load_gps_data(csv_path: str):
    """
    Loads GPS data from a CSV file with columns: 
    [timestamp, latitude, longitude].
    Returns a DataFrame with parsed timestamps.
    """
    df = pd.read_csv(csv_path)
    # Ensure 'timestamp' is a proper datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Sort by time in case it's not sorted
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# -------------------------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------------------------
def create_features(df: pd.DataFrame):
    """
    Given a DataFrame with [timestamp, latitude, longitude],
    create additional features.
    """
    # Calculate time difference (in seconds) between consecutive rows
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    # Calculate approximate distance traveled between consecutive points
    R = 6371_000  # Earth radius in meters
    lat_rad = np.radians(df['latitude'])
    lon_rad = np.radians(df['longitude'])
    dlat = lat_rad.diff().fillna(0)
    dlon = lon_rad.diff().fillna(0)
    lat_mean = (lat_rad + lat_rad.shift(1)) / 2
    lat_mean.fillna(lat_rad, inplace=True)
    dx = dlon * np.cos(lat_mean)
    dy = dlat
    df['distance_m'] = R * np.sqrt(dx**2 + dy**2)

    # Approximate speed (m/s)
    df['speed_m_s'] = df.apply(
        lambda row: row['distance_m'] / row['time_diff'] if row['time_diff'] != 0 else 0,
        axis=1
    )

    # Additional features: hour and day of week
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    df.fillna(0, inplace=True)
    return df

# -------------------------------------------------
# 3. BUILD A DATASET FOR TIME SERIES
# -------------------------------------------------
def create_supervised_sequences(df: pd.DataFrame, sequence_length=10):
    """
    Convert the DataFrame into a supervised learning problem.
    Uses the last `sequence_length` rows to predict the next coordinate.
    """
    feature_cols = ['latitude', 'longitude', 'speed_m_s', 'hour_of_day', 'day_of_week']
    target_cols = ['latitude', 'longitude']
    data = df[feature_cols].values
    targets = df[target_cols].values

    X, y = [], []
    for i in range(len(df) - sequence_length):
        seq_x = data[i : i + sequence_length]
        seq_y = targets[i + sequence_length]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)
    return X, y

# -------------------------------------------------
# 4. NORMALIZE FEATURES
# -------------------------------------------------
def normalize_data(X, y):
    """
    Scale input features to [0,1] range.
    Returns scaled X, scaled y, and the scalers.
    """
    num_samples, seq_len, num_features = X.shape
    X_2d = X.reshape(num_samples * seq_len, num_features)
    x_scaler = MinMaxScaler()
    X_2d_scaled = x_scaler.fit_transform(X_2d)
    X_scaled = X_2d_scaled.reshape(num_samples, seq_len, num_features)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)
    return X_scaled, y_scaled, x_scaler, y_scaler

# -------------------------------------------------
# 5. BUILD THE LSTM MODEL
# -------------------------------------------------
def build_lstm_model(input_shape):
    """
    Build a simple LSTM model with Keras.
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='linear'))  # Predict latitude and longitude
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# -------------------------------------------------
# 6. PREDICT FUTURE GPS COORDINATES
# -------------------------------------------------
def predict_future_coordinates(model, df, x_scaler, y_scaler, sequence_length=10, future_steps=1):
    """
    Predict the next 'future_steps' GPS coordinates using an auto-regressive approach.
    """
    feature_cols = ['latitude', 'longitude', 'speed_m_s', 'hour_of_day', 'day_of_week']
    last_window = df[feature_cols].values[-sequence_length:]
    last_window_2d = last_window.reshape(-1, len(feature_cols))
    last_window_2d_scaled = x_scaler.transform(last_window_2d)
    current_seq = last_window_2d_scaled.reshape(1, sequence_length, len(feature_cols))
    predictions = []

    for _ in range(future_steps):
        pred_scaled = model.predict(current_seq)
        pred = y_scaler.inverse_transform(pred_scaled)[0]
        predictions.append(pred.tolist())

        # Create new row for the predicted step
        new_time = df['timestamp'].iloc[-1] + pd.Timedelta(seconds=10)
        new_hour = new_time.hour
        new_day_of_week = new_time.dayofweek
        last_speed = df['speed_m_s'].iloc[-1]
        new_row = np.array([pred[0], pred[1], last_speed, new_hour, new_day_of_week])
        new_row_scaled = x_scaler.transform(new_row.reshape(1, -1))
        current_seq = np.concatenate([current_seq[:, 1:, :], new_row_scaled.reshape(1, 1, -1)], axis=1)

        # Append minimal info to df (for demonstration)
        new_row_df = {
            'timestamp': new_time, 'latitude': pred[0], 'longitude': pred[1],
            'time_diff': 0, 'distance_m': 0, 'speed_m_s': last_speed,
            'hour_of_day': new_hour, 'day_of_week': new_day_of_week
        }
        df = df.append(new_row_df, ignore_index=True)

    return predictions

# -------------------------------------------------
# Console Output Redirector
# -------------------------------------------------
class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
    def flush(self):
        pass

# -------------------------------------------------
# TKinter GUI Application
# -------------------------------------------------
class GPSPredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("GPS Prediction App")
        self.SEQ_LEN = 10

        # Initialize data and model variables
        self.df = None
        self.model = None
        self.x_scaler = None
        self.y_scaler = None

        self.create_widgets()

    def create_widgets(self):
        # Top frame for controls and data display
        frame_top = tk.Frame(self.master)
        frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Buttons frame
        frame_buttons = tk.Frame(frame_top)
        frame_buttons.pack(side=tk.TOP, fill=tk.X)

        tk.Button(frame_buttons, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(frame_buttons, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(frame_buttons, text="Predict Future", command=self.predict_future).pack(side=tk.LEFT, padx=5, pady=5)

        # Historical Data Viewer (Treeview)
        frame_data = tk.LabelFrame(frame_top, text="Historical Data")
        frame_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree = ttk.Treeview(frame_data)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame_data, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Future Predictions Display
        frame_pred = tk.LabelFrame(frame_top, text="Future Predictions")
        frame_pred.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.pred_text = tk.Text(frame_pred, height=10, width=40)
        self.pred_text.pack(fill=tk.BOTH, expand=True)

        # Console Output Area
        console_frame = tk.LabelFrame(self.master, text="Console Output")
        console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.console_text = tk.Text(console_frame, height=10)
        self.console_text.pack(fill=tk.BOTH, expand=True)
        sys.stdout = ConsoleRedirector(self.console_text)

    def load_csv(self):
        file_path = filedialog.askopenfilename(title="Select GPS CSV file", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            print(f"\nLoading data from {file_path}...\n")
            self.df = load_gps_data(file_path)
            print("Data loaded successfully.")
            self.df = create_features(self.df)
            self.update_treeview(self.df)

    def update_treeview(self, df):
        # Clear existing columns and data
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80)
        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=list(row))

    def train_model(self):
        if self.df is None:
            print("\nNo data loaded. Please load a CSV file first.\n")
            return
        # Run training in a separate thread to keep the GUI responsive
        threading.Thread(target=self._train_model_thread).start()

    def _train_model_thread(self):
        print("\nPreparing data for training...")
        X, y = create_supervised_sequences(self.df, sequence_length=self.SEQ_LEN)
        X_scaled, y_scaled, self.x_scaler, self.y_scaler = normalize_data(X, y)
        print("Building LSTM model...")
        self.model = build_lstm_model(input_shape=(X_scaled.shape[1], X_scaled.shape[2]))
        print("Training model... (this may take a moment)")
        self.model.fit(X_scaled, y_scaled, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
        print("Model training completed.")

    def predict_future(self):
        if self.df is None or self.model is None:
            print("\nData or model not available. Please load data and train the model first.\n")
            return
        print("\nPredicting future coordinates...")
        future_steps = 5
        predictions = predict_future_coordinates(self.model, self.df.copy(), self.x_scaler, self.y_scaler,
                                                   sequence_length=self.SEQ_LEN, future_steps=future_steps)
        self.pred_text.delete("1.0", tk.END)
        self.pred_text.insert(tk.END, "Predicted future GPS coordinates:\n")
        for i, coords in enumerate(predictions, start=1):
            pred_str = f"Step {i}: Latitude={coords[0]:.6f}, Longitude={coords[1]:.6f}\n"
            self.pred_text.insert(tk.END, pred_str)
        print("Future predictions completed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GPSPredictionApp(root)
    root.mainloop()
