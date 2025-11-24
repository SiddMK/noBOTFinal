# unified_retrieval.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class UnifiedRetriever:
    """
    Enhanced unified retrieval system for searching across multiple datasets
    using semantic embeddings.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the retriever with datasets and embeddings.
        
        Args:
            base_path: Base directory path. If None, uses script location.
        """
        # Setup paths
        if base_path is None:
            base_path = Path(__file__).parent
        else:
            base_path = Path(base_path)
        
        self.base_path = base_path
        self.data_dir = base_path / "data"
        self.embeddings_dir = self.data_dir / "embeddings"
        
        print("🔄 Loading unified retriever...")
        
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  ✓ Loaded embedding model")
        
        # Define file mappings
        self.csv_files = {
            "students": "noBOT_dataset_cleaned.csv",
            "engagement": "Engagement_dataset_cleaned.csv",
            "timetable": "timetable_dataset.csv",
            "events": "dataset_events1.csv",
            "college_map": "dataset_roadmap_cleaned.csv"
        }
        
        self.embedding_files = {
            "students": "students_embeddings.npy",
            "engagement": "engagement_embeddings.npy",
            "timetable": "timetable_embeddings.npy",
            "events": "events_embeddings.npy",
            "college_map": "roadmap_embeddings.npy"
        }
        
        # Load all data
        self._load_data()
    
    def _load_data(self):
        """Load all datasets and embeddings with error handling."""
        self.datasets = {}
        self.embeddings = {}
        
        print("\n📂 Loading datasets...")
        for name, filename in self.csv_files.items():
            try:
                csv_path = self.data_dir / filename
                
                # Special handling for timetable
                if name == "timetable":
                    self.datasets[name] = pd.read_csv(
                        csv_path, 
                        encoding="latin1", 
                        engine="python"
                    )
                else:
                    self.datasets[name] = pd.read_csv(csv_path)
                
                print(f"  ✓ Loaded {name}: {len(self.datasets[name])} rows")
                
            except FileNotFoundError:
                print(f"  ⚠️  Warning: {filename} not found, skipping...")
            except Exception as e:
                print(f"  ❌ Error loading {name}: {e}")
                raise
        
        print("\n🧮 Loading embeddings...")
        for name, filename in self.embedding_files.items():
            try:
                emb_path = self.embeddings_dir / filename
                self.embeddings[name] = np.load(emb_path)
                print(f"  ✓ Loaded {name} embeddings: {self.embeddings[name].shape}")
                
            except FileNotFoundError:
                print(f"  ⚠️  Warning: {filename} not found, skipping...")
            except Exception as e:
                print(f"  ❌ Error loading {name} embeddings: {e}")
                raise
        
        # Validate matching datasets and embeddings
        self._validate_data()
        print("\n✅ Unified retriever loaded successfully.\n")
    
    def _validate_data(self):
        """Ensure datasets and embeddings are aligned."""
        for name in self.datasets.keys():
            if name not in self.embeddings:
                raise ValueError(f"Missing embeddings for dataset: {name}")
            
            expected_rows = len(self.datasets[name])
            actual_embs = len(self.embeddings[name])
            
            if expected_rows != actual_embs:
                raise ValueError(
                    f"Mismatch in {name}: "
                    f"{expected_rows} rows but {actual_embs} embeddings"
                )
    
    @lru_cache(maxsize=100)
    def embed_query(self, query: str):
        """
        Embed user query with caching for repeated queries.
        
        Args:
            query: Search query string
            
        Returns:
            numpy array of query embedding
        """
        return self.model.encode([query], convert_to_numpy=True)
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.0) -> List[Dict]:
        """
        Search across all datasets and return top_k results.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of dicts with dataset, score, and row data
        """
        query_embedding = self.embed_query(query)
        all_results = []

        # Collect results from all datasets
        for name, df in self.datasets.items():
            embs = self.embeddings[name]
            scores = cosine_similarity(query_embedding, embs)[0]
            
            # Get ALL scores with indices
            for idx, score in enumerate(scores):
                if score >= threshold:
                    all_results.append({
                        "dataset": name,
                        "score": float(score),
                        "row": df.iloc[idx].to_dict(),
                        "index": idx
                    })
        
        # Sort by score descending
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return all_results[:top_k]
    
    def search_dataset(
        self, 
        query: str, 
        dataset_name: str, 
        top_k: int = 3, 
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search within a specific dataset only.
        
        Args:
            query: Search query string
            dataset_name: Name of dataset to search
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching results from the specified dataset
        """
        if dataset_name not in self.datasets:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self.datasets.keys())}"
            )
        
        query_embedding = self.embed_query(query)
        df = self.datasets[dataset_name]
        embs = self.embeddings[dataset_name]
        
        scores = cosine_similarity(query_embedding, embs)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score >= threshold:
                results.append({
                    "dataset": dataset_name,
                    "score": float(score),
                    "row": df.iloc[idx].to_dict(),
                    "index": idx
                })
        
        return results
    
    def format_result(self, result: Dict) -> str:
        """
        Format a single result for display.
        
        Args:
            result: Result dictionary
            
        Returns:
            Formatted string
        """
        lines = [
            f"📊 Dataset: {result['dataset'].upper()}",
            f"🎯 Similarity: {result['score']:.2%}",
            f"\n📄 Data:"
        ]
        
        for key, value in result['row'].items():
            lines.append(f"  • {key}: {value}")
        
        return "\n".join(lines)
    
    def format_results(self, results: List[Dict]) -> str:
        """
        Format multiple results for display.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Formatted string with all results
        """
        if not results:
            return "❌ No results found."
        
        output = [f"Found {len(results)} result(s):\n"]
        
        for i, result in enumerate(results, 1):
            output.append(f"\n{'='*60}")
            output.append(f"RESULT #{i}")
            output.append('='*60)
            output.append(self.format_result(result))
        
        return "\n".join(output)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about loaded data.
        
        Returns:
            Dictionary with statistics for each dataset
        """
        return {
            name: {
                "rows": len(df),
                "columns": list(df.columns),
                "embedding_shape": self.embeddings[name].shape
            }
            for name, df in self.datasets.items()
        }
    
    def print_stats(self):
        """Print dataset statistics in a formatted way."""
        stats = self.get_stats()
        print("\n📊 DATASET STATISTICS:")
        print("="*60)
        for name, info in stats.items():
            print(f"\n{name.upper()}:")
            print(f"  Rows: {info['rows']}")
            print(f"  Embeddings: {info['embedding_shape']}")
            print(f"  Columns: {', '.join(info['columns'][:5])}" + 
                  ("..." if len(info['columns']) > 5 else ""))
        print("\n" + "="*60)


# ------------------------------------------------
# Enhanced testing interface
# ------------------------------------------------
if __name__ == "__main__":
    # Initialize with custom path
    base_path = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT"
    
    try:
        retriever = UnifiedRetriever(base_path)
    except Exception as e:
        print(f"\n❌ Failed to initialize retriever: {e}")
        exit(1)
    
    # Show available datasets
    print("\n" + "="*60)
    print("📚 Available datasets:", ", ".join(retriever.datasets.keys()))
    print("="*60)
    
    # Show statistics
    retriever.print_stats()
    
    # Interactive loop
    while True:
        print("\n" + "="*60)
        print("OPTIONS:")
        print("  1. Search all datasets")
        print("  2. Search specific dataset")
        print("  3. Show statistics")
        print("  4. Exit")
        print("="*60)
        
        choice = input("\nChoice: ").strip()
        
        if choice == "4":
            print("\n👋 See you soon!")
            break
        
        elif choice == "3":
            retriever.print_stats()
            continue
        
        query = input("\n🔍 Enter query: ").strip()
        if not query:
            print("⚠️  Query cannot be empty!")
            continue
        
        try:
            if choice == "1":
                top_k_input = input("How many results? (default 3): ").strip()
                top_k = int(top_k_input) if top_k_input else 3
                
                threshold_input = input("Minimum similarity (0-1, default 0.0): ").strip()
                threshold = float(threshold_input) if threshold_input else 0.0
                
                print("\n🔄 Searching...")
                results = retriever.search(query, top_k=top_k, threshold=threshold)
                print("\n" + retriever.format_results(results))
                
            elif choice == "2":
                print(f"\nAvailable datasets: {list(retriever.datasets.keys())}")
                dataset = input("Dataset name: ").strip()
                
                top_k_input = input("How many results? (default 3): ").strip()
                top_k = int(top_k_input) if top_k_input else 3
                
                print("\n🔄 Searching...")
                results = retriever.search_dataset(query, dataset, top_k=top_k)
                print("\n" + retriever.format_results(results))
            
            else:
                print("⚠️  Invalid choice!")
                
        except ValueError as e:
            print(f"\n❌ Error: {e}")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")