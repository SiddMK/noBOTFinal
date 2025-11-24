import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
#  PART 1 — LOAD, PROCESS, ENCODE & SAVE EMBEDDINGS
# =========================================================

def build_and_save_embeddings():
    print("🔹 Loading dataset...")
    df = pd.read_csv("dataset_events1.csv")   # raw CSV in your current folder

    print("🔹 Creating context column...")
    df["context"] = (
        df["Event_Name"].astype(str) + " " +
        df["Description"].astype(str) + " " +
        df["Organizing_Department"].astype(str) + " " +
        df["Venue"].astype(str)
    )

    print("🔹 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔹 Generating embeddings...")
    embeddings = model.encode(
        df["context"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Save processed file
    df.to_csv(
        r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\events_with_context.csv",
        index=False
    )

    # Save embedding matrix
    np.save(
        r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\events_embeddings.npy",
        embeddings
    )

    print("✅ Embeddings and processed dataset saved successfully!")



# =========================================================
#  PART 2 — RAG SEARCH FUNCTION
# =========================================================

def load_rag_components():
    print("🔹 Loading processed dataset...")
    df = pd.read_csv(
        r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\events_with_context.csv"
    )

    print("🔹 Loading embeddings...")
    embeddings = np.load(
        r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\events_embeddings.npy"
    )

    print("🔹 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return df, embeddings, model



def search_event(query, df, embeddings, model, top_k=1):
    print(f"\n🔍 Searching for: {query}")

    # Encode user question
    query_vec = model.encode([query])

    # Calculate similarity
    scores = cosine_similarity(query_vec, embeddings)[0]

    # Get best match
    best = scores.argsort()[::-1][:top_k]

    result = df.iloc[best[0]]

    return {
        "event_name": result["Event_Name"],
        "description": result["Description"],
        "date": result["Date"],
        "time": result["Time"],
        "venue": result["Venue"],
        "similarity": float(scores[best[0]])
    }



# =========================================================
#  RUN BOTH PARTS
# =========================================================

if __name__ == "__main__":
    print("⚡ Running noBOT Events RAG Pipeline...")

    # STEP 1 — Build embeddings (run once)
    build_and_save_embeddings()

    # STEP 2 — Load data + search system
    df, embeddings, model = load_rag_components()

    # TEST QUERY
    test_query = "tech hackathon in my college"
    answer = search_event(test_query, df, embeddings, model)

    print("\n🎯 TOP MATCH:")
    print(answer)
