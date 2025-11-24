import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load engagement dataset
df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\Engagement_dataset_cleaned.csv")

# Possible columns that may exist in your engagement dataset
possible_cols = [
    "Student_Name",
    "Faculty_Name",
    "Subject",
    "Interaction_Type",
    "Remarks",
    "Department",
    "Section",
    "Semester",
    "Engagement_Level"
]

# Detect only available columns
used_cols = [col for col in possible_cols if col in df.columns]

print("Using columns for ENGAGEMENT context:", used_cols)

# Build context string
df["context"] = df[used_cols].astype(str).agg(" ".join, axis=1)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(df["context"].tolist(), show_progress_bar=True)

# Save embeddings
np.save(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\embeddings\engagement_embeddings.npy", embeddings)

print("Engagement embeddings saved successfully!")
print("Shape:", embeddings.shape)
