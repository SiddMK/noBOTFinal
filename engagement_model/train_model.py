import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load CSV
df = pd.read_csv("Engagement_dataset-final.csv")


# Clean missing numeric values
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Encode 'Class_advisor'
le = LabelEncoder()
df['Class_advisor'] = le.fit_transform(df['Class_advisor'])

# Separate features and target
X = df.drop(columns=['Engagement_category'])
y = df['Engagement_category']

# Force numeric conversion
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(numeric_only=True), inplace=True)


# Scale numeric columns
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include='number').columns
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=1)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")
