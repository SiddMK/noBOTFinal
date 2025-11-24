# =============================
# noBOT Student Performance Model
# =============================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: hides long TensorFlow CPU logs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# STEP 1 — Load Dataset
df = pd.read_csv("../data/noBOT_dataset_cleaned.csv")

# STEP 2 — Select Features (X) and Target (y)
X = df[['Age_scaled', 'Gpa_scaled', 'Attendance_scaled', 'Absences_scaled',
        'Study_hrs_perday', 'Math_scaled', 'Science_scaled']]

y = df['Performance_category']

# STEP 3 — Encode Target Labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# STEP 4 — Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# STEP 5 — Define Neural Network Architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: High, Medium, Low
])

# STEP 6 — Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# STEP 7 — Train the Model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# STEP 8 — Evaluate Performance
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model Accuracy on Test Data: {acc*100:.2f}%")

# STEP 9 — Save the Model & Encoder
os.makedirs("../models", exist_ok=True)
model.save("../models/noBOT_student_performance_model.h5")

import joblib
joblib.dump(encoder, "../models/label_encoder.pkl")

print("\n🎯 Model and encoder saved successfully!")
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
