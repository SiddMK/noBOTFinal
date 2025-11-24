import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your processed events CSV
df = pd.read_csv(
    r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\dataset_events1.csv"
)

# Create the text column to embed
df["context_text"] = df.apply(
    lambda row: f"{row['event_name']} - {row['event_description']} on {row['date']} at {row['time']}",
    axis=1
)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(df["context_text"].tolist(), show_progress_bar=True)

# Save embeddings
np.save(
    r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\events_embeddings.npy",
    embeddings
)
