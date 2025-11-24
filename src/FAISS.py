import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle

# Load your timetable dataset
data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\src\timetable_sentences_fixed.csv")

# Convert each row into a text sentence
sentences = data.apply(
    lambda row: f"Section {row['Section']} has {row['Subject']} with {row['Faculty']} at {row['Time']}.",
    axis=1
).tolist()

# Generate embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(sentences, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Save embeddings and sentences
with open(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\timetable_embeddings.pkl", "wb") as f:
    pickle.dump({"sentences": sentences, "embeddings": embeddings}, f)

print("✅ Timetable embeddings created and saved successfully!")
