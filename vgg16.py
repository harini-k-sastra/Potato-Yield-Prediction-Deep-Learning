#VGG16
# ============================================
# Step 1: Import Libraries
# ============================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# ============================================
# Step 2: Load Dataset
# ============================================
data = pd.read_csv('Potato.csv', encoding='latin1')


# ============================================
# Step 3: Define Features and Target
# ============================================
target_column = 'hg/ha_yield'

categorical_cols = ['ï»¿Area', 'Item']
numerical_cols = [
    'Year',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp'
]

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
# Step 7: Convert Tabular → Image (VGG16 Trick)
# ============================================
num_features = X_scaled.shape[1]

target_size_flat = 32 * 32  # 1024
X_padded = np.zeros((X_scaled.shape[0], target_size_flat))
X_padded[:, :num_features] = X_scaled

# Reshape
X_image = X_padded.reshape(-1, 32, 32, 1)
X_image = np.repeat(X_image, 3, axis=-1)

# ============================================
# Step 8: Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_image, y_scaled, test_size=0.2, random_state=42
)

# ============================================
# Step 9: Load VGG16
# ============================================
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# ============================================
# Step 10: Custom Layers
# ============================================
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='linear')(x)

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
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# ============================================
# Step 13: Evaluate + R2
# ============================================
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

loss, mae = model.evaluate(X_test, y_test)
print("Test Loss (MSE):", loss)
print("Test MAE (Scaled):", mae)

# Predictions
y_pred = model.predict(X_test)

# Convert back to original scale
y_test_actual = scaler_y.inverse_transform(y_test)
y_pred_actual = scaler_y.inverse_transform(y_pred)

# Metrics

r2 = r2_score(y_test_actual, y_pred_actual)


print(f"R2 Score: {r2 * 100:.2f}%")

# ============================================
# Step 14: Save
# ============================================
model.save('vgg16_potato_model.h5')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
joblib.dump(X.columns.tolist(), 'X_train_columns.pkl')




#VGG 16 PREDICTION
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# ============================
# Load Model
# ============================
model = load_model('vgg16_potato_model.h5', compile=False)
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

# Ensure same order
sample_df_encoded = sample_df_encoded[X_train_columns]

# ============================
# Step 5: Scale Input
# ============================
sample_scaled = scaler_X.transform(sample_df_encoded)

# ============================
# 🔥 Step 6: Convert to VGG16 Input (IMPORTANT)
# ============================
target_size_flat = 32 * 32  # 1024

# Pad
sample_padded = np.zeros((1, target_size_flat))
sample_padded[:, :sample_scaled.shape[1]] = sample_scaled

# Reshape to image
sample_image = sample_padded.reshape(1, 32, 32, 1)

# Convert to 3 channels
sample_image = np.repeat(sample_image, 3, axis=-1)

# ============================
# Step 7: Predict
# ============================
prediction_scaled = model.predict(sample_image)

# ============================
# Step 8: Inverse Transform
# ============================
prediction = scaler_y.inverse_transform(prediction_scaled)

print("🌾 Predicted Potato Yield:", prediction[0][0])