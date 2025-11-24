"""
noBOT: Smart Campus Assistant - Fixed Version with RAG
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import google.generativeai as genai

# Suppress progress bars and warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedRetriever:
    """Smart campus assistant for navigating and understanding college campus."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize the retriever with datasets and embeddings."""
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.data_dir = self.base_path / "data"
        self.embeddings_dir = self.data_dir / "embeddings"
        
        logger.info("Loading noBOT Smart Campus Assistant...")
        
        # Load embedding model with progress bar suppression
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        self.model.max_seq_length = 128
        
        # File mappings
        self.files = {
            "students": ("noBOT_dataset_cleaned.csv", "students_embeddings.npy", "Student Information"),
            "engagement": ("Engagement_dataset_cleaned.csv", "engagement_embeddings.npy", "Class Engagement & Attendance"),
            "timetable": ("timetable_dataset.csv", "timetable_embeddings.npy", "Class Schedules"),
            "events": ("dataset_events1.csv", "events_embeddings.npy", "Events & Programs"),
            "college_map": ("dataset_roadmap_cleaned.csv", "roadmap_embeddings.npy", "Campus Locations")
        }
        
        # Dataset keywords for intelligent routing
        self.keywords = {
            "students": ["student", "admission", "grade", "course", "major", "department", "enroll"],
            "engagement": ["attendance rate", "engagement", "participation", "assignment completion"],
            "timetable": ["schedule", "timetable", "class", "lecture", "period", "time", "timing"],
            "events": ["event", "fest", "festival", "competition", "program", "cultural", "technical"],
            "college_map": ["where", "location", "find", "building", "room", "lab", "library", "canteen", 
                           "office", "department", "block", "directions"]
        }
        
        self.datasets = {}
        self.embeddings = {}
        self._load_data()
    
    def _load_data(self):
        """Load all datasets and embeddings."""
        for name, (csv_file, emb_file, _) in self.files.items():
            try:
                csv_path = self.data_dir / csv_file
                encoding = "latin1" if name == "timetable" else "utf-8"
                self.datasets[name] = pd.read_csv(csv_path, encoding=encoding, engine="python")
                
                emb_path = self.embeddings_dir / emb_file
                self.embeddings[name] = np.load(emb_path)
                
                if len(self.datasets[name]) != len(self.embeddings[name]):
                    raise ValueError(f"Data mismatch in {name}")
                    
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                raise
        
        logger.info("Ready to help! Ask me anything about the campus.")
    
    @lru_cache(maxsize=100)
    def embed_query(self, query: str) -> np.ndarray:
        """Embed user query with progress bar suppressed."""
        return self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    
    def extract_filters(self, query: str) -> Dict[str, str]:
        """Extract filtering criteria from query."""
        filters = {}
        query_lower = query.lower()
        
        section_match = re.search(r'\b(?:section|class)\s+([a-z])\b', query_lower)
        if section_match:
            filters['section'] = section_match.group(1).upper()
        
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day in query_lower:
                filters['day'] = day.capitalize()
                break
        
        return filters
    
    def detect_intent(self, query: str) -> Optional[str]:
        """Detect which dataset is most relevant."""
        query_lower = query.lower()
        
        if any(t in query_lower for t in ["attendance rate", "engagement rate", "assignment completion"]):
            return "engagement"
        
        if "attendance" in query_lower and not any(w in query_lower for w in ["schedule", "timetable"]):
            return "engagement"
        
        if ("list" in query_lower or "show" in query_lower) and "student" in query_lower:
            return "students"
        
        if any(t in query_lower for t in ["when is", "details about"]) and \
           any(w in query_lower for w in ["hackathon", "fest", "event", "workshop"]):
            return "events"
        
        if "schedule" in query_lower or "timetable" in query_lower:
            return "timetable"
        
        if any(t in query_lower for t in ["where is", "find", "location", "how to reach"]):
            return "college_map"
        
        scores = {dataset: sum(2 for kw in keywords if kw in query_lower) 
                  for dataset, keywords in self.keywords.items()}
        
        best = max(scores, key=scores.get) if scores else None
        return best if best and scores[best] >= 2 else None
    
    def apply_filters(self, df: pd.DataFrame, filters: Dict, dataset: str) -> pd.DataFrame:
        """Apply filters to dataframe."""
        if df.empty or not filters:
            return df
        
        result = df.copy()
        
        if "section" in filters:
            col = 'Section' if 'Section' in result.columns else 'Class' if 'Class' in result.columns else None
            if col:
                result = result[result[col].astype(str).str.upper() == filters["section"]]
        
        if dataset == "timetable" and "day" in filters and 'Day' in result.columns:
            result = result[result['Day'].astype(str).str.lower() == filters["day"].lower()]
        
        if dataset == "engagement" and "section" in filters:
            class_col = f"Class {filters['section']}"
            if class_col in result.columns:
                result = result[result[class_col] == 1]
        
        return result
    
    def search_dataset(self, query: str, dataset: str, top_k: int = 3, 
                      threshold: float = 0.3, filters: Optional[Dict] = None) -> List[Dict]:
        """Search within a specific dataset."""
        df = self.datasets.get(dataset)
        if df is None or df.empty:
            return []
        
        if filters:
            df = self.apply_filters(df, filters, dataset)
            if df.empty:
                return []
        
        if filters and len(df) < len(self.datasets[dataset]):
            embs = self.embeddings[dataset][df.index.tolist()]
        else:
            embs = self.embeddings[dataset]
        
        query_emb = self.embed_query(query)
        scores = cosine_similarity(query_emb, embs)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [{"dataset": dataset, "score": float(scores[idx]), 
                 "row": df.iloc[idx].to_dict(), "index": int(df.index[idx])}
                for idx in top_indices if scores[idx] >= threshold]
    
    def check_quick_answers(self, query: str) -> Optional[str]:
        """Check for common questions with direct answers."""
        q = query.lower()
        
        if any(w in q for w in ['cafeteria', 'canteen', 'food', 'eat', 'lunch', 'dining']):
            return """**Campus Dining Options:**

🍽️ **Main Cafeteria** - Block B, Ground Floor
⏰ 8:00 AM - 7:00 PM (Mon-Sat)
Full meals, snacks, and beverages available

🍵 **Staff Canteen** - Admin Block, First Floor  
⏰ 9:00 AM - 5:00 PM

🍕 **Food Court** - Student Center
⏰ 10:00 AM - 8:00 PM
Pizza, burgers, and South Indian food"""
        
        if 'club' in q and not any(w in q for w in ['music club', 'coding club', 'when', 'where']):
            return """**Student Clubs & Activities:**

💻 **TECHNICAL[GDG Club]:** Coding (Wed 5-7PM), Robotics (Tue 4-6PM), AI/ML (Fri 5-7PM)
🎭 **CULTURAL:** Music[DHWANI] (Fri 4-6PM), Drama[ACTKRIT] (Thu 4-6PM), Dance[SYNAPSE] (Sat 3-5PM)
📸 **CREATIVE:** Photography (Sat 3-5PM), Art (Sun 2-4PM)
⚽ **SPORTS[RSC]:** Daily 5-7 PM

📧 Contact: club@college.edu"""
        
        return None
    
    def smart_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Intelligently search by detecting intent."""
        quick = self.check_quick_answers(query)
        if quick:
            return [{"dataset": "quick_answer", "score": 1.0, "row": {"answer": quick}, "index": 0}]
        
        filters = self.extract_filters(query)
        target = self.detect_intent(query)
        
        logger.info(f"Query: '{query}' | Dataset: {target} | Filters: {filters}")
        
        configs = {
            "timetable": (10, 0.15),
            "engagement": (5, 0.2),
            "college_map": (3, 0.25),
            "events": (3, 0.25),
            "students": (5, 0.25)
        }
        
        if target and target in configs:
            k, thresh = configs[target]
            results = self.search_dataset(query, target, top_k=k, threshold=thresh, filters=filters)
            if results:
                return results[:top_k]
        
        query_emb = self.embed_query(query)
        all_results = []
        
        for name, df in self.datasets.items():
            if df.empty:
                continue
            scores = cosine_similarity(query_emb, self.embeddings[name])[0]
            for idx, score in enumerate(scores):
                if score >= 0.25:
                    all_results.append({
                        "dataset": name, "score": float(score),
                        "row": df.iloc[idx].to_dict(), "index": idx
                    })
        
        return sorted(all_results, key=lambda x: x["score"], reverse=True)[:top_k]
 
    def format_result(self, result: Dict) -> str:
        """Format a single result."""
        row = result['row']
        dataset = result['dataset']
        category = self.files.get(dataset, (None, None, dataset))[2]
        
        lines = [f"**{category}:**\n"]
        
        if dataset == 'events':
            lines.extend([
                f"📅 **Event:** {row.get('Event_Name', 'N/A')}",
                f"🏷️ **Type:** {row.get('Event_Type', 'N/A')}",
                f"📆 **Date:** {row.get('Date', 'N/A')} at {row.get('Time', 'N/A')}",
                f"📍 **Venue:** {row.get('Venue', 'N/A')}"
            ])
            if row.get('Description'):
                lines.append(f"ℹ️ **Details:** {row['Description']}")
            if row.get('Registration_Link'):
                lines.append(f"🔗 **Register:** {row['Registration_Link']}")
        
        elif dataset == 'timetable':
            lines.extend([
                f"📚 **Subject:** {row.get('Subject', 'N/A')}",
                f"👥 **Section:** {row.get('Section') or row.get('Class', 'N/A')}",
                f"📅 **Day:** {row.get('Day', 'N/A')}",
                f"⏰ **Time:** {row.get('Time', 'N/A')}",
                f"🏫 **Room:** {row.get('Venue', 'N/A')}",
                f"👨‍🏫 **Instructor:** {row.get('Faculty', 'N/A')}"
            ])
        
        elif dataset == 'college_map':
            location = ' - '.join([p for p in [row.get('Room_Name'), row.get('Department'), 
                                              row.get('Building_Name')] if p and str(p) != '-'])
            lines.append(f"📍 **Location:** {location or 'N/A'}")
            
            if row.get('Block_Code') and str(row['Block_Code']) != '-':
                lines.append(f"🏢 **Block:** {row['Block_Code']}")
            if row.get('Floor_No') is not None:
                floor = "Ground Floor" if row['Floor_No'] == 0 else f"Floor {row['Floor_No']}"
                lines.append(f"🔢 **Floor:** {floor}")
            if row.get('Faculty_Name') and str(row['Faculty_Name']) != '-':
                lines.append(f"👤 **Contact Person:** {row['Faculty_Name']}")
                if row.get('Designation') and str(row['Designation']) != '-':
                    lines.append(f"💼 **Designation:** {row['Designation']}")
            if row.get('Contact') and str(row['Contact']) != '-':
                lines.append(f"📧 **Email:** {row['Contact']}")
            if row.get('Landmark'):
                lines.append(f"🗺️ **Nearby:** {row['Landmark']}")
            if row.get('Special_Guidance'):
                lines.append(f"ℹ️ **Note:** {row['Special_Guidance']}")
        
        elif dataset == 'engagement':
            class_name = next((col.replace('Class ', '') for col in ['Class A', 'Class B', 'Class C', 
                              'Class D', 'Class E', 'Class F'] if col in row and row[col] == 1), None)
            if class_name:
                lines.append(f"👥 **Class:** {class_name}")
            if row.get('Class_advisor'):
                lines.append(f"👨‍🏫 **Advisor:** {row['Class_advisor']}")
            if row.get('Attendance_rate-scale') is not None:
                lines.append(f"📊 **Attendance:** {float(row['Attendance_rate-scale']) * 100:.1f}%")
            if row.get('Assignment_rate-scale') is not None:
                lines.append(f"📝 **Assignments:** {float(row['Assignment_rate-scale']) * 100:.1f}%")
            if row.get('Participation_score') is not None:
                lines.append(f"🎯 **Participation:** {float(row['Participation_score']) * 100:.1f}%")
        
        return "\n".join(lines)
    
    def format_response(self, results: List[Dict]) -> str:
        """Format complete response."""
        if not results:
            return "I couldn't find specific information about that. Try asking about events, schedules, locations, clubs, or dining options."
        
        if results[0].get('dataset') == 'quick_answer':
            return results[0]['row']['answer']
        
        if len(results) == 1:
            return self.format_result(results[0])
        
        response = []
        for i, result in enumerate(results, 1):
            response.append(self.format_result(result))
            if i < len(results):
                response.append("\n---\n")
        return "\n".join(response)


class RAGPipeline:
    """RAG Pipeline integrating retrieval with LLM generation."""
    
    def __init__(self, retriever: UnifiedRetriever, api_key: Optional[str] = None):
        """Initialize RAG pipeline with retriever and optional API key."""
        self.retriever = retriever
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('AIzaSyBnBn1uq_x61qE93OKcs9JO-ECifKTE6ik')
        
        if not self.api_key:
            logger.warning("No Gemini API key found. Using retrieval-only mode.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                # FIXED: Use correct model name
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini LLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.model = None
    
    def ask(self, query: str) -> str:
        """Main method to handle queries with RAG."""
        try:
            # Get relevant documents
            results = self.retriever.smart_search(query, top_k=3)
            
            # Handle quick answers
            if results and results[0].get('dataset') == 'quick_answer':
                return results[0]['row']['answer']
            
            # Format context from retrieval
            context = self.retriever.format_response(results)
            
            # If no LLM available, return formatted context
            if not self.model or not results:
                return context if context else "I couldn't find information about that. Please try rephrasing your question."
            
            # Generate LLM response with context
            prompt = f"""You are noBOT, a friendly and helpful campus assistant for RNS Institute of Technology.

Student Question: {query}

Retrieved Information:
{context}

Instructions:
- Provide a natural, conversational response based ONLY on the information above
- Be concise (2-4 sentences) but informative
- If the information doesn't fully answer the question, say so politely
- Maintain a friendly, helpful tone
- Use the structured data naturally in your response"""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            # Fallback to retrieval-only response
            try:
                results = self.retriever.smart_search(query, top_k=2)
                return self.retriever.format_response(results)
            except Exception as fallback_error:
                logger.error(f"Fallback error: {fallback_error}")
                return "I encountered an error processing your request. Please try again."


def main():
    """Main entry point for CLI usage."""
    import sys
    
    # Get base path from environment or default to current directory
    base_path = os.getenv('NOBOT_DATA_PATH', os.path.dirname(__file__))
    
    try:
        print("Initializing noBOT...")
        retriever = UnifiedRetriever(base_path)
        
        # Get API key from environment
        api_key = os.getenv('AIzaSyBnBn1uq_x61qE93OKcs9JO-ECifKTE6ik')
        if not api_key:
            print("\n⚠️  Warning: GEMINI_API_KEY not found in environment variables.")
            print("Running in retrieval-only mode. Set GEMINI_API_KEY for LLM responses.\n")
        
        rag = RAGPipeline(retriever, api_key=api_key)
        
        print("\n" + "="*60)
        print("       Welcome to noBOT - Your Campus Assistant")
        print("="*60)
        print("\nAsk me about: events, schedules, locations, clubs, dining, and more!")
        print("Commands: 'exit' to quit\n")
        
        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("\nThank you for using noBOT! Have a great day!\n")
                    break
                
                print("\n🤖 noBOT: ", end="")
                response = rag.ask(query)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print("Sorry, something went wrong. Please try again.\n")
    
    except Exception as e:
        print(f"\n❌ Error initializing noBOT: {e}")
        print("Please check your data files and paths.\n")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())