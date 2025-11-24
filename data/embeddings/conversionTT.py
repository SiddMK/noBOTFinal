import pandas as pd
import numpy as np
import pickle

# Load the PKL file
with open(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\embeddings\roadmap_gemini_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

# Extract embeddings column
embeddings = np.vstack(df["embedding"].values)

# Save as .npy file
np.save(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\embeddings\roadmap_embeddings.npy", embeddings)

print("Timetable embeddings saved successfully!")
print("Shape:", embeddings.shape)
