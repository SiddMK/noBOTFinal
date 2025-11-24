import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# -----------------------------------------
# PATHS (Update if required)
# -----------------------------------------
DATA_DIR = r"C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\embeddings"
CSV_FILE = "timetable_dataset.csv"      # <- Your corrected timetable file
EMB_PATH = os.path.join(DATA_DIR, "embeddings", "timetable_embeddings.npy")

# Make sure embeddings folder exists
os.makedirs(os.path.join(DATA_DIR, "embeddings"), exist_ok=True)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
print("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------
# LOAD CSV
# -----------------------------------------
print("📄 Reading timetable CSV...")
df = pd.read_csv(os.path.join(DATA_DIR, CSV_FILE))

# Columns used for embedding (edit if needed)
TEXT_COLUMNS = ["Day", "Time", "Subject", "Faculty", "Section"]

# Combine text from selected columns
def combine_text(row):
    return " | ".join(str(row[col]) for col in TEXT_COLUMNS if col in row)

texts = df.apply(combine_text, axis=1).tolist()

# -----------------------------------------
# GENERATE EMBEDDINGS
# -----------------------------------------
print(f"⚙️ Generating embeddings for timetable... ({len(texts)} rows)")
embeddings = model.encode(texts, convert_to_numpy=True)

# -----------------------------------------
# SAVE
# -----------------------------------------
np.save(EMB_PATH, embeddings)

print("✅ Timetable embeddings generated and saved at:")
print(EMB_PATH)
