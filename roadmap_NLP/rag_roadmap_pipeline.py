# ==========================================
# noBOT RAG Embedding Script (Gemini version)
# ==========================================
# Purpose: Generate embeddings for roadmap dataset using Gemini
# Output : roadmap_gemini_embeddings.csv
# Author  : Siddharth M Kulkarni
# ==========================================

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm

# ---------------------------
# 1️⃣ Configure Gemini API Key
# ---------------------------
# Get your key from Google AI Studio: https://makersuite.google.com/app/apikey
genai.configure(api_key="AIzaSyDJJi0munxtv0jOPPukCeXcQ2t0XM1FNAE")

# ---------------------------
# 2️⃣ Load Your Clean Dataset
# ---------------------------
df = pd.read_csv("dataset_roadmap_cleaned.csv")

# ---------------------------
# 3️⃣ Convert Each Row → Text
# ---------------------------
def row_to_text(row):
    """Convert a row into a descriptive natural-language chunk."""
    return (
        f"{row['Faculty_Name']}, {row['Designation']} at {row['Department']} department, "
        f"sits in {row['Room_Name']} ({row['Room_Type']}) located in {row['Building_Name']} "
        f"Block {row['Block_Code']}, Floor {row['Floor_No']}. "
        f"Landmark: {row['Landmark']}. Notes: {row['Notes']}. "
        f"Special Guidance: {row['Special_Guidance']}."
    )

# Apply conversion
df["text_chunk"] = df.apply(row_to_text, axis=1)

# ---------------------------
# 4️⃣ Generate Gemini Embeddings
# ---------------------------
model = "models/embedding-001"

def get_gemini_embedding(text):
    """Generate embedding using Gemini's embedding model."""
    try:
        result = genai.embed_content(model=model, content=text)
        return result["embedding"]
    except Exception as e:
        print(f"⚠️ Error embedding text: {e}")
        return None

# Apply with progress bar for better tracking
tqdm.pandas(desc="Generating Gemini Embeddings")
df["embedding"] = df["text_chunk"].progress_apply(get_gemini_embedding)

# ---------------------------
# 5️⃣ Save Output File
# ---------------------------
# Option 1: Save as CSV (for inspection)
df.to_csv("roadmap_gemini_embeddings.csv", index=False)

# Option 2: Save as Pickle (for direct use with FAISS)
df.to_pickle("roadmap_gemini_embeddings.pkl")

print("\n✅ Embeddings successfully generated and saved!")
