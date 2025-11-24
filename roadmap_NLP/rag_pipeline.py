import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ✅ 1. Initialize the embedder (same one used during embedding generation)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ 2. Configure Gemini
genai.configure(api_key="AIzaSyDJJi0munxtv0jOPPukCeXcQ2t0XM1FNAE")
model = genai.GenerativeModel(model_name="models/gemini-pro-latest")


# ✅ 3. Load saved embeddings
with open(r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\src\timetable_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

sentences = data["sentences"]
embeddings = np.array(data["embeddings"])

# ✅ 4. Retrieval
def retrieve(query, top_k=3):
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [sentences[i] for i in top_indices]

# ✅ 5. Answer generation
def generate_answer(query):
    context = "\n".join(retrieve(query))
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

# ✅ 6. Example query
query = "What is the timetable for Section B on Monday?"
answer = generate_answer(query)
print("\n🧠 Answer:\n", answer)
