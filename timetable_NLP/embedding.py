
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load your timetable sentences
df = pd.read_csv("timetable_sentences_fixed.csv")

# Load a pretrained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Generate embeddings
embeddings = model.encode(df['Sentence'].tolist(), show_progress_bar=True)

# Add embeddings to dataframe
df['embedding'] = embeddings.tolist()

# Save for retrieval
df.to_pickle("timetable_with_embeddings.pkl")

print("✅ Embeddings successfully generated and saved!")
