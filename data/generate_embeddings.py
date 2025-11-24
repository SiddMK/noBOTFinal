# generate_embeddings.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 1️⃣ Load dataset
df = pd.read_csv("dataset_events1.csv")

# 2️⃣ Combine important text columns into one "context"
df["context"] = (
    df["Event_Name"].astype(str) + " " +
    df["Description"].astype(str) + " " +
    df["Organizing_Department"].astype(str) + " " +
    df["Venue"].astype(str)
)

# 3️⃣ Load pre-trained model for sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4️⃣ Generate embeddings
embeddings = model.encode(
    df["context"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

# 5️⃣ Save embeddings for later use
np.save(
    r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\events_embeddings.npy",
    embeddings
)

# 6️⃣ Save processed dataset
df.to_csv(
    r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\events_with_context.csv",
    index=False
)

print("Embeddings generated and saved successfully.")
print(f"Shape of embeddings: {embeddings.shape}")
