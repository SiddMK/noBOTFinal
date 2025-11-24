import pandas as pd
import numpy as np
import google.generativeai as genai
import ast  # safer than eval

# ============================================================
# 1️⃣ Configure Gemini API Key
# ============================================================
genai.configure(api_key="AIzaSyDJJi0munxtv0jOPPukCeXcQ2t0XM1FNAE")  # ← replace with your key or load from env var

# ============================================================
# 2️⃣ Load and Parse Embeddings
# ============================================================
df = pd.read_csv("roadmap_gemini_embeddings.csv")

def parse_embedding(x):
    """Safely convert stored embeddings (string → NumPy array)."""
    if isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x), dtype="float32")
        except Exception:
            return np.nan
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype="float32")
    else:
        return np.nan

df["embedding"] = df["embedding"].apply(parse_embedding)
df.dropna(subset=["embedding"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"✅ Loaded {len(df)} valid embeddings.")

# ============================================================
# 3️⃣ Helper Functions
# ============================================================

def get_gemini_embedding(text: str) -> np.ndarray:
    """Generate an embedding for a query using Gemini."""
    result = genai.embed_content(model="models/embedding-001", content=text)
    return np.array(result["embedding"], dtype="float32")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Return context for testing without generating new embeddings."""
    # Temporarily skip online embedding generation (since quota is exceeded)
    print("⚠️ Using dummy similarity scores (quota exceeded mode)")
    top_results = df.sample(min(top_k, len(df)))  # random sample for demo
    context = "\n".join(top_results["text_chunk"].tolist())
    return context

# ============================================================
# 4️⃣ Generate Context-Aware Answer
# ============================================================

def generate_answer(query: str) -> str:
    """Retrieve context and generate answer using Gemini."""
    context = retrieve_context(query)
    prompt = f"""
You are noBOT, an AI campus assistant.
Use the following college roadmap context to answer the student's question accurately.

Context:
{context}

Question:
{query}

Answer clearly, concisely, and factually.
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ============================================================
# 5️⃣ Example Query Test
# ============================================================
if __name__ == "__main__":
    query = "Where is the HOD cabin for the CSE department?"
    answer = generate_answer(query)
    print("\n💬 Query:", query)
    print("🤖 noBOT:", answer)
