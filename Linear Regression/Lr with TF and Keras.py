import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import plotly.express as px
from tensorflow.keras import layers, regularizers

# 1. READ DATASET
print("--- Loading Data ---")
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print(f'Total number of rows: {len(training_df.index)}')
print(training_df.head()) # simplified head for terminal

# 2. DATA STATISTICS (Printing to console)
print("\n--- Statistics ---")
print(training_df.describe(include='all'))

# 3. VERIFY ANSWERS (Standard Python Logic)
print("\n--- Insights ---")
max_fare = training_df['FARE'].max()
mean_distance = training_df['TRIP_MILES'].mean()
num_unique_companies = training_df['COMPANY'].nunique()
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
missing_values = training_df.isnull().sum().sum()

print(f"Max Fare: ${max_fare:.2f}")
print(f"Mean Distance: {mean_distance:.4f} miles")
print(f"Cab Companies: {num_unique_companies}")
print(f"Most Freq Payment: {most_freq_payment_type}")
print(f"Missing Data: {'No' if missing_values == 0 else 'Yes'}")

# 4. VISUALIZATION
# This will open a browser window with the chart when run locally
print("\n--- Generating Plot ---")
fig = px.scatter_matrix(training_df.sample(500), dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
fig.show()

# 5. DEFINE ML MODEL (Refactored to remove ml_edu dependency)
# We replaced the 'ExperimentSettings' object with standard arguments

def create_model(input_features, learning_rate):
    
    # 1. Create Inputs (Same as before)
    inputs = {name: keras.Input(shape=(1,), name=name) for name in input_features}
    
    # 2. Concatenate (Same as before)
    concatenated_inputs = layers.Concatenate()(list(inputs.values()))

    # ---------------------------------------------------------
    # NEW: FEATURE SCALING (Normalization)
    # ---------------------------------------------------------
    # This layer automatically centers your data (Mean = 0, Std Dev = 1).
    # It replaces the need to use StandardScaler in Scikit-Learn.
    # Note: You must call model.layers[-2].adapt(data) before training!
    x = layers.Normalization(axis=-1, name="Normalizer")(concatenated_inputs)

    # ---------------------------------------------------------
    # NEW: REGULARIZATION & FEATURE SELECTION
    # ---------------------------------------------------------
    # kernel_regularizer: Applies a penalty on the weights.
    # 'l1': (Lasso) Sets useless feature weights to exactly 0. (Feature Selection)
    # 'l2': (Ridge) Shrinks weights close to 0 but not exactly. (Prevents Overfitting)
    # 'l1_l2': Combines both strategies.
    outputs = layers.Dense(units=1, 
                           name='Label',
                           kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01) 
                          )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=["mean_squared_error"])
    return model

def train_model(model, dataset, input_features, label_name, batch_size=32, epochs=10):
    """Train the model by feeding it data."""
    
    # Separate features and labels
    features = {name: dataset[name].values for name in input_features}
    label = dataset[label_name].values

    print(f"\n--- Starting Training for {epochs} epochs ---")
    history = model.fit(x=features,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)
    return history

# 6. EXECUTE THE TRAINING
# Let's try to predict FARE based on TRIP_MILES
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "FARE"
input_features = ["TRIP_MILES"]

# Create and Train
my_model = create_model(input_features, learning_rate)
epochs_history = train_model(my_model, training_df, input_features, label_name, batch_size, epochs)

print("\n--- Training Complete ---")

learning_rate = 0.001
epochs = 20
batch_size = 50
input_features = ['TRIP_MILES', 'TRIP_MINUTES']

training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60
twofeatures_my_model = create_model(input_features, learning_rate)
twofeaturesepochs_history = train_model(twofeatures_my_model, training_df, input_features, label_name, batch_size, epochs)

def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return


output = predict_fare(twofeatures_my_model, training_df, input_features, 'FARE')
show_predictions(output)