from sentence_transformers import SentenceTransformer
import pandas as pd

# Load your sentences
df = pd.read_csv("timetable_sentences_fixed.csv")

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
sentences = df['Sentence'].dropna().tolist()
embeddings = model.encode(sentences)

print(f"✓ Generated {len(embeddings)} embeddings!")