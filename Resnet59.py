#Resnet 59
# ============================================
# 1. Import Libraries
# ============================================
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.layers import Input, Dense, Add, Activation, BatchNormalization
from tensorflow.keras.models import Model

# ============================================
# 2. Load Dataset
# ============================================
data = pd.read_csv("Potato.csv", encoding='latin1')
data.columns = data.columns.str.replace('ï»¿', '')

# ============================================
# 3. Define Features & Target
# ============================================
target_column = 'hg/ha_yield'
categorical_cols = ['Area', 'Item']

data_encoded = pd.get_dummies(data, columns=categorical_cols)

X = data_encoded.drop(columns=[target_column])
y = data_encoded[target_column]

# ============================================
# 4. Handle Missing Values
# ============================================
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# ============================================
# 5. Scaling
# ============================================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# ============================================
# 6. Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ============================================
# 7. Residual Block
# ============================================
def residual_block(x, units):
    shortcut = x

    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(units)(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# ============================================
# 8. Build Model (ResNet-like)
# ============================================
input_dim = X_train.shape[1]

inputs = Input(shape=(input_dim,))

x = Dense(128, activation='relu')(inputs)

for _ in range(48):   # depth (ResNet-like)
    x = residual_block(x, 128)

x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

outputs = Dense(1)(x)

model = Model(inputs, outputs)

# ============================================
# 9. Compile Model
# ============================================
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ============================================
# 10. Train Model
# ============================================
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16
)

# ============================================
# 11. Evaluate Model
# ============================================
loss, mae = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
loss, mae = model.evaluate(X_test, y_test)
print("Test Loss (MSE):", loss)
print("Test MAE (Scaled):", mae)

# inverse transform
y_test_actual = scaler_y.inverse_transform(y_test)
y_pred_actual = scaler_y.inverse_transform(y_pred)

# metrics
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"R2 Score: {r2 * 100:.2f}%")
# ============================================
# Save Model + Preprocessors
# ============================================
model.save("custom_resnet_model.h5")

import joblib
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
joblib.dump(X.columns.tolist(), "X_train_columns.pkl")

#RESNET 59 PREDICTION
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ============================
# Load trained model
# ============================
model = load_model("custom_resnet_model.h5", compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ============================
# Load preprocessors
# ============================
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
X_train_columns = joblib.load("X_train_columns.pkl")

# ============================
# New input data
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
# Convert to DataFrame
# ============================
sample_df = pd.DataFrame([sample])

# ============================
# One-hot encoding
# ============================
sample_df_encoded = pd.get_dummies(sample_df)

# ============================
# Align columns
# ============================
missing_cols = set(X_train_columns) - set(sample_df_encoded.columns)

for col in missing_cols:
    sample_df_encoded[col] = 0

sample_df_encoded = sample_df_encoded[X_train_columns]

# ============================
# Scale input
# ============================
sample_scaled = scaler_X.transform(sample_df_encoded)

# ============================
# Predict
# ============================
prediction_scaled = model.predict(sample_scaled)

# ============================
# Inverse transform
# ============================
prediction = scaler_y.inverse_transform(prediction_scaled)

print("🌾 Predicted Potato Yield:", prediction[0][0])