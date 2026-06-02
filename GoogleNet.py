    #GoogleNet
       # ============================================
# Step 1: Import Libraries
# ============================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# ============================================
# Step 2: Load Dataset
# ============================================
data = pd.read_csv('Potato.csv', encoding='latin1')
data.columns = data.columns.str.replace('ï»¿', '')
# ============================================
# Step 3: Define Features and Target
# ============================================
target_column = 'hg/ha_yield'
categorical_cols = ['Area', 'Item']

# ============================================
# Step 4: One-Hot Encoding
# ============================================
data_encoded = pd.get_dummies(data, columns=categorical_cols)

X = data_encoded.drop(columns=[target_column])
y = data_encoded[target_column]

# ============================================
# Step 5: Handle Missing Values
# ============================================
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# ============================================
# Step 6: Scaling
# ============================================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# ============================================
# Step 7: Tabular → Image + Resize
# ============================================
import tensorflow as tf

num_features = X_scaled.shape[1]
target_size = 32 * 32

X_padded = np.zeros((X_scaled.shape[0], target_size))
X_padded[:, :num_features] = X_scaled

X_image = X_padded.reshape(-1, 32, 32, 1)
X_image = np.repeat(X_image, 3, axis=-1)

# Resize to 75x75 (required for Inception)
X_image = tf.image.resize(X_image, (75, 75)).numpy()

# ============================================
# Step 8: Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_image, y_scaled, test_size=0.2, random_state=42
)

# ============================================
# Step 9: Load InceptionV3 (GoogLeNet)
# ============================================
from tensorflow.keras.applications import InceptionV3

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(75, 75, 3)
)

for layer in base_model.layers:
     layer.trainable = False

# ============================================
# Step 10: Custom Layers
# ============================================
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=base_model.input, outputs=output)

# ============================================
# Step 11: Compile
# ============================================
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ============================================
# Step 12: Train
# ============================================
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

# ============================================
# Step 13: Evaluate + R2
# ============================================
loss, mae = model.evaluate(X_test, y_test)
print("Test Loss (MSE):", loss)
print("Test MAE (Scaled):", mae)


y_pred = model.predict(X_test)

y_test_actual = scaler_y.inverse_transform(y_test)
y_pred_actual = scaler_y.inverse_transform(y_pred)

r2 = r2_score(y_test_actual, y_pred_actual)

print(f"R2 Score: {r2 * 100:.2f}%")

# ============================================
# Step 14: Save
# ============================================
model.save('inception_potato_model.h5')


# ============================
# GoogLeNet (InceptionV3) PREDICTION
# ============================

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================
# Load Model
# ============================
model = load_model('inception_potato_model.h5', compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ============================
# Load Preprocessors
# ============================
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
X_train_columns = joblib.load('X_train_columns.pkl')

# ============================
# Step 1: New Input
# ============================
sample = {
    'Area': 'Albania',
    'Item': 'Potatoes',
    'Year': 2030,
    'average_rain_fall_mm_per_year': 1485,
    'pesticides_tonnes': 121,
    'avg_temp': 22
}

# ============================
# Step 2: Convert to DataFrame
# ============================
sample_df = pd.DataFrame([sample])

# ============================
# Step 3: One-Hot Encoding
# ============================
categorical_cols = ['Area', 'Item']
sample_df_encoded = pd.get_dummies(sample_df, columns=categorical_cols)

# ============================
# Step 4: Align Columns
# ============================
missing_cols = set(X_train_columns) - set(sample_df_encoded.columns)

for col in missing_cols:
    sample_df_encoded[col] = 0

sample_df_encoded = sample_df_encoded[X_train_columns]

# ============================
# Step 5: Scale Input
# ============================
sample_scaled = scaler_X.transform(sample_df_encoded)

# ============================
# Step 6: Convert to GoogLeNet Input
# ============================
target_size_flat = 32 * 32  # 1024

sample_padded = np.zeros((1, target_size_flat))
sample_padded[:, :sample_scaled.shape[1]] = sample_scaled

sample_image = sample_padded.reshape(1, 32, 32, 1)

# Convert to 3 channels (RGB)
sample_image = np.repeat(sample_image, 3, axis=-1)

# Resize to 75x75 (InceptionV3 requirement)
sample_image = tf.image.resize(sample_image, (75, 75)).numpy()

# ============================
# Step 7: Predict
# ============================
prediction_scaled = model.predict(sample_image)

# ============================
# Step 8: Inverse Transform
# ============================
prediction = scaler_y.inverse_transform(prediction_scaled)

print("🌾 Predicted Potato Yield:", prediction[0][0])