import pandas as pd
import tensorflow as tf
import joblib

# Load model and encoder
model = tf.keras.models.load_model("../models/noBOT_student_performance_model.h5")
encoder = joblib.load("../models/label_encoder.pkl")

# Example new student data
new_student = pd.DataFrame([{
    'Age_scaled': 0.5,
    'Gpa_scaled': 0.8,
    'Attendance_scaled': 0.9,
    'Absences_scaled': 0.1,
    'Study_hrs_perday': 0.7,
    'Math_scaled': 0.75,
    'Science_scaled': 0.82
}])

# Predict
pred = model.predict(new_student)
pred_class = encoder.inverse_transform([pred.argmax()])[0]

print(f" Predicted Performance Category: {pred_class}")
