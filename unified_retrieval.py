# unified_retrieval.py

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class UnifiedRetriever:
    def __init__(self):
        print("🔄 Loading unified retriever...")

        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # -------------------------------
        # Load CSV datasets
        # -------------------------------
        self.datasets = {
            "students": pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\noBOT_dataset_cleaned.csv"),
            "engagement": pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\Engagement_dataset_cleaned.csv"),
            "timetable": pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\timetable_dataset.csv",encoding="latin1",engine="python"),
            "events": pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\dataset_events1.csv"),
            "college_map": pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\dataset_roadmap_cleaned.csv")
        }

        # -------------------------------
        # Load NPY embeddings
        # -------------------------------
        self.embeddings = {
            "students": np.load("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\embeddings\\students_embeddings.npy"),
            "engagement": np.load("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\embeddings\\engagement_embeddings.npy"),
            "timetable": np.load("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\embeddings\\timetable_embeddings.npy"),
            "events": np.load("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\embeddings\\events_embeddings.npy"),
            "college_map": np.load("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\embeddings\\roadmap_embeddings.npy")
        }

        print("✅ Unified retriever loaded successfully.\n")

    # ------------------------------------------------
    # Embed user query
    # ------------------------------------------------
    def embed_query(self, query):
        return self.model.encode([query], convert_to_numpy=True)

    # ------------------------------------------------
    # Find top match across ALL datasets
    # ------------------------------------------------
    def search(self, query, top_k=1):
        query_embedding = self.embed_query(query)

        best_result = None
        best_score = -1

        # Search each dataset
        for name, df in self.datasets.items():
            embs = self.embeddings[name]

            scores = cosine_similarity(query_embedding, embs)[0]  # 1 x N

            top_index = np.argmax(scores)
            top_score = scores[top_index]

            if top_score > best_score:
                best_score = top_score
                best_result = {
                    "dataset": name,
                    "score": float(top_score),
                    "row": df.iloc[top_index].to_dict()
                }

        return best_result


# ------------------------------------------------
# For manual testing
# ------------------------------------------------
if __name__ == "__main__":
    retriever = UnifiedRetriever()

    while True:
        q = input("\nAsk something: ")
        result = retriever.search(q)
        print("\n🎯 TOP MATCH:")
        print(result)
