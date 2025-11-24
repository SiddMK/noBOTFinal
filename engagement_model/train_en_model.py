import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# 1️⃣ Load dataset
df = pd.read_excel("Engagement_dataset_cleaned.xlsx")

# 2️⃣ Auto-encode all non-numeric columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"✅ Encoded column: {col} | Unique values: {len(le.classes_)}")

# 3️⃣ Define features and target
if 'Engagement_Level' not in df.columns:
    raise KeyError("❌ 'Engagement_Level' column not found in dataset!")

X = df.drop(columns=['Engagement_Level'])
y = df['Engagement_Level']

# 4️⃣ Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6️⃣ Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 categories: Low, Medium, High
])

# 7️⃣ Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# 8️⃣ Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,           # you can increase this safely now
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)


# 9️⃣ Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Model Accuracy: {acc:.2f}")

# 🔟 Plot accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
