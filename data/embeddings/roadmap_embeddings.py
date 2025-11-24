import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

CSV_PATH = r"C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\embeddings\\dataset_roadmap_cleaned.csv"
OUTPUT_EMB = r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\embeddings\timetable_embeddings.npy"

TEXT_COLUMN = "combined_text"   # <-- CHANGE THIS IF YOUR CSV DOES NOT HAVE THIS

# Load CSV
df = pd.read_csv(CSV_PATH)

# If combined_text doesn't exist, create it manually
if TEXT_COLUMN not in df.columns:
    # Create a combined text field using all columns
    df[TEXT_COLUMN] = df.astype(str).agg(" ".join, axis=1)
    print("⚠️ 'combined_text' not found — created automatically by merging ALL columns.")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode
embeddings = model.encode(df[TEXT_COLUMN].tolist(), show_progress_bar=True)

# Save as .npy
np.save(OUTPUT_EMB, embeddings)

print("Embeddings saved to:", OUTPUT_EMB)
