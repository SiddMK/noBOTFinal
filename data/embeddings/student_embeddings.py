import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\noBOT_dataset_cleaned.csv")

# ---- BUILD CONTEXT SAFELY ----
possible_cols = [
    "Name",
    "USN",
    "Section",
    "Email",
    "Phone",
    "Department",
    "Semester",
    "Class_Advisor"   # ← ADDED THIS
]

# Use only the columns that actually exist
used_cols = [col for col in possible_cols if col in df.columns]

print("Using columns for context:", used_cols)

# Create context string by joining available columns
df["context"] = df[used_cols].astype(str).agg(" ".join, axis=1)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(df["context"].tolist(), show_progress_bar=True)

# Save embeddings
np.save(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\embeddings\students_embeddings.npy", embeddings)

print("Students embeddings saved successfully!")
print("Shape:", embeddings.shape)
