import chromadb
import pandas as pd

# Load your embeddings
df = pd.read_pickle("timetable_with_embeddings.pkl")

# 🧹 Drop rows where 'Sentence' or 'embedding' is missing
df = df.dropna(subset=['Sentence', 'embedding'])

# 🧾 Convert to string just in case
df['Sentence'] = df['Sentence'].astype(str)

# Initialize Chroma client
client = chromadb.Client()

# Create or access collection
collection = client.get_or_create_collection("timetable")

# Add cleaned data
collection.add(
    documents=df['Sentence'].tolist(),
    embeddings=df['embedding'].tolist(),
    ids=[str(i) for i in range(len(df))]
)

print("✅ Embeddings added to ChromaDB successfully!")
# 🔍 Test query on your timetable collection

query = "When does Section D have Python Lab?"
results = collection.query(
    query_texts=[query],
    n_results=1  # get top 3 closest matches
)

for i, doc in enumerate(results['documents'][0]):
    print(f"\nResult {i+1}:")
    print(doc)
