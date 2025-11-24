# unified_retrieval.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


class UnifiedRetriever:
    """
    noBOT: Smart Campus Assistant
    A user-friendly system for answering campus-related questions.
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
        
        print("Loading noBOT Smart Campus Assistant...")
        print("Please wait a moment...\n")
        
        # Load embedding model (silently)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
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
        
        # User-friendly dataset descriptions
        self.dataset_info = {
            "students": "Student Information",
            "engagement": "Activities & Clubs",
            "timetable": "Class Schedules",
            "events": "Events & Programs",
            "college_map": "Campus Locations"
        }
        
        # Dataset keywords for smart routing
        self.dataset_keywords = {
            "students": ["student", "admission", "grade", "course", "major", "department", "enroll"],
            "engagement": ["activity", "club", "participation", "workshop", "interest", "join"],
            "timetable": ["schedule", "timetable", "class", "lecture", "period", "time", "when", "timing"],
            "events": ["event", "fest", "festival", "competition", "program", "function", "cultural", "technical"],
            "college_map": ["where", "location", "find", "map", "building", "room", "lab", "library", "canteen", "direction", "way"]
        }
        
        # Load all data
        self._load_data()
    
    def _load_data(self):
        """Load all datasets and embeddings with error handling."""
        self.datasets = {}
        self.embeddings = {}
        
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
                
            except FileNotFoundError:
                print(f"Warning: {filename} not found")
            except Exception as e:
                print(f"Error loading {name}: {e}")
                raise
        
        for name, filename in self.embedding_files.items():
            try:
                emb_path = self.embeddings_dir / filename
                self.embeddings[name] = np.load(emb_path)
                
            except FileNotFoundError:
                print(f"Warning: {filename} embeddings not found")
            except Exception as e:
                print(f"Error loading {name} embeddings: {e}")
                raise
        
        # Validate matching datasets and embeddings
        self._validate_data()
        print("Ready to help! Ask me anything about the campus.\n")
    
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
        """Embed user query with caching."""
        return self.model.encode([query], convert_to_numpy=True)
    
    def detect_intent(self, query: str) -> Optional[str]:
        """
        Intelligently detect which dataset is most relevant to the query.
        
        Args:
            query: User query string
            
        Returns:
            Dataset name or None for global search
        """
        query_lower = query.lower()
        
        # Count keyword matches for each dataset
        scores = {}
        for dataset, keywords in self.dataset_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[dataset] = score
        
        # If clear winner exists, return it
        if scores:
            best_dataset = max(scores, key=scores.get)
            if scores[best_dataset] >= 2:  # Require at least 2 keyword matches
                return best_dataset
        
        return None  # Global search
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict]:
        """Search across all datasets."""
        query_embedding = self.embed_query(query)
        all_results = []

        for name, df in self.datasets.items():
            embs = self.embeddings[name]
            scores = cosine_similarity(query_embedding, embs)[0]
            
            for idx, score in enumerate(scores):
                if score >= threshold:
                    all_results.append({
                        "dataset": name,
                        "score": float(score),
                        "row": df.iloc[idx].to_dict(),
                        "index": idx
                    })
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
    
    def search_dataset(
        self, 
        query: str, 
        dataset_name: str, 
        top_k: int = 3, 
        threshold: float = 0.3
    ) -> List[Dict]:
        """Search within a specific dataset only."""
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
    
    def check_quick_answers(self, query: str) -> Optional[str]:
        """
        Check for common questions that can be answered directly.
        Returns formatted answer if found, None otherwise.
        """
        query_lower = query.lower()
        
        # Cafeteria information - check if query is about food/dining
        cafeteria_keywords = ['cafeteria', 'canteen', 'food', 'eat', 'lunch', 'dinner', 'breakfast', 'cafe', 'dining', 'mess']
        if any(word in query_lower for word in cafeteria_keywords):
            return """
[Campus Dining]
--------------------------------------------------

Main Cafeteria
Location: Block B, Ground Floor
Timings: 8:00 AM - 7:00 PM (Monday to Saturday)
Features: Full meals, snacks, beverages
How to get there: Enter main gate, turn right. Next to Sports Ground.

Staff Canteen
Location: Admin Block, First Floor
Timings: 9:00 AM - 5:00 PM
Features: Tea, coffee, light snacks
Note: Primarily for faculty and staff

Food Court
Location: Student Center, Ground Floor
Timings: 10:00 AM - 8:00 PM
Features: Multiple food stalls, pizza, burgers, south Indian food
Popular spot for: Group hangouts and quick bites
"""
        
        # Clubs information - more aggressive detection
        # Check if the query is specifically about clubs/activities
        clubs_indicators = [
            'club' in query_lower,
            'extracurricular' in query_lower,
            ('student' in query_lower and 'activity' in query_lower),
            ('student' in query_lower and 'activities' in query_lower),
            'what clubs' in query_lower,
            'which clubs' in query_lower,
            'join club' in query_lower,
            'about clubs' in query_lower,
            'list' in query_lower and 'club' in query_lower,
            'all clubs' in query_lower,
            'available clubs' in query_lower,
            'tell me about club' in query_lower,
            'details about club' in query_lower,
            'information about club' in query_lower,
            'what activities' in query_lower,
            'which activities' in query_lower
        ]
        
        # If any indicator is true, show clubs list
        if any(clubs_indicators):
            # Don't show clubs list if asking about a specific club name or specific event
            # (let the regular search handle specific queries)
            specific_queries = ['music club', 'coding club', 'drama club', 'dance club', 
                              'robotics club', 'photo', 'art club', 'sports club',
                              'when does', 'where does', 'who runs', 'coordinator']
            
            # If not asking about specifics, show the full list
            if not any(specific in query_lower for specific in specific_queries):
                return """
[Student Clubs & Activities]
--------------------------------------------------

TECHNICAL CLUBS:
• Coding Club - Programming, hackathons, tech workshops
  Meets: Wednesday 5-7 PM | Contact: coding@college.edu

• Robotics Club - Build robots, automation projects  
  Meets: Tuesday 4-6 PM | Contact: robotics@college.edu

• AI/ML Club - Machine learning, data science projects
  Meets: Friday 5-7 PM | Contact: aiml@college.edu

CULTURAL CLUBS:
• Music Club - Instruments, singing, band performances
  Meets: Friday 4-6 PM | Contact: music@college.edu

• Drama Club - Theatre, acting, plays
  Meets: Thursday 4-6 PM | Contact: drama@college.edu

• Dance Club - Various dance forms and performances
  Meets: Saturday 3-5 PM | Contact: dance@college.edu

CREATIVE CLUBS:
• Photography Club - Photo walks, editing, exhibitions
  Meets: Saturday 3-5 PM | Contact: photo@college.edu

• Art Club - Painting, sketching, creative arts
  Meets: Sunday 2-4 PM | Contact: art@college.edu

SPORTS & FITNESS:
• Sports Club - Cricket, football, athletics
  Meets: Daily 5-7 PM | Contact: sports@college.edu

To join any club, contact them directly or visit the Student Affairs Office.
"""
        
        return None
    
    def smart_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Intelligently search by detecting intent and routing to appropriate dataset.
        """
        # First check for quick answers
        quick_answer = self.check_quick_answers(query)
        if quick_answer:
            # Return as a special result format
            return [{
                "dataset": "quick_answer",
                "score": 1.0,
                "row": {"answer": quick_answer},
                "index": 0
            }]
        
        # Detect most relevant dataset
        target_dataset = self.detect_intent(query)
        
        if target_dataset:
            # Search specific dataset
            results = self.search_dataset(query, target_dataset, top_k=top_k, threshold=0.25)
            if results:
                return results
        
        # Fallback to global search
        return self.search(query, top_k=top_k, threshold=0.25)
    
    def format_result_simple(self, result: Dict, show_category: bool = True) -> str:
        """
        Format a single result in a simple, readable way for all users.
        
        Args:
            result: Result dictionary
            show_category: Whether to show the category/source
            
        Returns:
            Formatted string in plain, clear language
        """
        row = result['row']
        lines = []
        
        # Add category header if needed
        if show_category:
            category = self.dataset_info.get(result['dataset'], result['dataset'])
            lines.append(f"\n[{category}]")
            lines.append("-" * 50)
        
        # Format based on dataset type - using simple, clear language
        if result['dataset'] == 'events':
            lines.append(f"Event Name: {row.get('Event_Name', 'Not specified')}")
            lines.append(f"Type: {row.get('Event_Type', 'Not specified')}")
            lines.append(f"Date: {row.get('Date', 'Not specified')}")
            lines.append(f"Time: {row.get('Time', 'Not specified')}")
            lines.append(f"Venue: {row.get('Venue', 'Not specified')}")
            
            if 'Description' in row and row['Description']:
                lines.append(f"Details: {row['Description']}")
            
            if 'Registration_Link' in row and row['Registration_Link']:
                lines.append(f"Register at: {row['Registration_Link']}")
            
            if 'Contact_Info' in row and row['Contact_Info']:
                lines.append(f"Contact: {row['Contact_Info']}")
        
        elif result['dataset'] == 'timetable':
            lines.append(f"Subject: {row.get('Subject', 'Not specified')}")
            lines.append(f"Day: {row.get('Day', 'Not specified')}")
            lines.append(f"Time: {row.get('Time', 'Not specified')}")
            lines.append(f"Room: {row.get('Venue', 'Not specified')}")
            
            if 'Faculty' in row and row['Faculty']:
                lines.append(f"Instructor: {row['Faculty']}")
        
        elif result['dataset'] == 'college_map':
            lines.append(f"Location: {row.get('Location', 'Not specified')}")
            
            if 'Building' in row and row['Building']:
                lines.append(f"Building: {row['Building']}")
            
            if 'Floor' in row and row['Floor']:
                lines.append(f"Floor: {row['Floor']}")
            
            if 'Directions' in row and row['Directions']:
                lines.append(f"How to get there: {row['Directions']}")
            
            if 'Landmarks' in row and row['Landmarks']:
                lines.append(f"Nearby: {row['Landmarks']}")
        
        elif result['dataset'] == 'engagement':
            if 'Club_Name' in row:
                lines.append(f"Club: {row.get('Club_Name', 'Not specified')}")
            if 'Activity_Name' in row:
                lines.append(f"Activity: {row.get('Activity_Name', 'Not specified')}")
            
            if 'Description' in row and row['Description']:
                lines.append(f"About: {row['Description']}")
            
            if 'Meeting_Time' in row and row['Meeting_Time']:
                lines.append(f"Meets: {row['Meeting_Time']}")
            
            if 'Contact' in row and row['Contact']:
                lines.append(f"Contact: {row['Contact']}")
        
        else:
            # Generic display for students or other datasets
            # Show first 5 most important fields
            important_fields = ['Name', 'Title', 'Department', 'Email', 'Phone']
            displayed = 0
            
            for field in important_fields:
                if field in row and row[field]:
                    lines.append(f"{field}: {row[field]}")
                    displayed += 1
            
            # If not enough important fields, show others
            if displayed < 3:
                for key, value in row.items():
                    if key not in important_fields and value and displayed < 5:
                        lines.append(f"{key}: {value}")
                        displayed += 1
        
        return "\n".join(lines)
    
    def format_response(self, results: List[Dict], query: str) -> str:
        """
        Format search results as a natural, friendly response.
        Uses simple language that anyone can understand.
        """
        if not results:
            return (
                "\nI couldn't find any information about that.\n\n"
                "You can ask me about:\n"
                "  - Upcoming events and programs\n"
                "  - Class schedules and timings\n"
                "  - Campus buildings and locations\n"
                "  - Student clubs and activities\n"
                "  - Cafeteria and dining options\n\n"
                "Try asking in a different way, or type 'help' for examples."
            )
        
        # Check if it's a quick answer
        if results[0].get('dataset') == 'quick_answer':
            return results[0]['row']['answer']
        
        # Build response
        response_lines = []
        
        if len(results) == 1:
            response_lines.append("\nHere's what I found:")
            response_lines.append(self.format_result_simple(results[0], show_category=True))
        else:
            response_lines.append(f"\nI found {len(results)} results for you:")
            for i, result in enumerate(results, 1):
                response_lines.append(f"\n--- Result {i} ---")
                response_lines.append(self.format_result_simple(result, show_category=True))
        
        return "\n".join(response_lines)
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded data."""
        return {
            name: {
                "rows": len(df),
                "columns": list(df.columns),
                "embedding_shape": self.embeddings[name].shape
            }
            for name, df in self.datasets.items()
        }


# ------------------------------------------------
# Simple, friendly interface for all users
# ------------------------------------------------
def show_welcome():
    """Display a simple, welcoming message."""
    print("\n" + "="*60)
    print("       Welcome to noBOT - Your Campus Assistant")
    print("="*60)
    print("\nHello! I'm here to help you find information about the campus.")
    print("\nYou can ask me questions like:")
    print("  - When is the cultural festival?")
    print("  - Where is the library?")
    print("  - What time is my next class?")
    print("  - Tell me about student clubs")
    print("  - Where can I get food?")
    print("  - How do I get to the cafeteria?")
    print("\nJust type your question naturally and press Enter.")
    print("\nSpecial commands:")
    print("  - Type 'help' to see this message again")
    print("  - Type 'examples' for more question examples")
    print("  - Type 'exit' when you're done")
    print("="*60 + "\n")


def show_examples():
    """Show more detailed examples."""
    print("\n" + "="*60)
    print("           Question Examples")
    print("="*60)
    print("\nAbout Events:")
    print("  - What events are happening this month?")
    print("  - When is the tech fest?")
    print("  - Tell me about upcoming competitions")
    print("\nAbout Schedules:")
    print("  - What is my class schedule?")
    print("  - When do I have mathematics class?")
    print("  - Show me today's timetable")
    print("\nAbout Campus:")
    print("  - Where can I find the computer lab?")
    print("  - How do I get to the auditorium?")
    print("  - Where is the student affairs office?")
    print("\nAbout Dining:")
    print("  - Where is the cafeteria?")
    print("  - What are the canteen timings?")
    print("  - Where can I get food on campus?")
    print("\nAbout Activities:")
    print("  - What clubs can I join?")
    print("  - List all available clubs")
    print("  - Tell me about the coding club")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Initialize
    base_path = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT"
    
    try:
        retriever = UnifiedRetriever(base_path)
    except Exception as e:
        print(f"\nSorry, there was a problem starting the system: {e}")
        print("Please contact technical support.\n")
        exit(1)
    
    show_welcome()
    
    # Main conversation loop
    while True:
        try:
            # Get user input with a friendly prompt
            query = input("Ask me anything: ").strip()
            
            if not query:
                continue
            
            # Handle special commands
            query_lower = query.lower()
            
            if query_lower in ['exit', 'quit', 'bye', 'goodbye', 'done']:
                print("\nThank you for using noBOT! Have a great day!\n")
                break
            
            elif query_lower in ['help', 'h', '?']:
                show_welcome()
                continue
            
            elif query_lower in ['examples', 'example', 'sample', 'samples']:
                show_examples()
                continue
            
            elif query_lower in ['stats', 'statistics', 'info', 'information']:
                stats = retriever.get_stats()
                print("\n" + "="*60)
                print("           System Information")
                print("="*60)
                for name, info in stats.items():
                    category = retriever.dataset_info.get(name, name)
                    print(f"\n{category}: {info['rows']} records available")
                print("\n" + "="*60 + "\n")
                continue
            
            # Perform smart search
            print("\nSearching for you...\n")
            results = retriever.smart_search(query, top_k=2)
            
            # Display results in simple language
            response = retriever.format_response(results, query)
            print(response)
            print()  # Extra line for readability
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\nOops! Something went wrong: {e}")
            print("Please try asking your question again.\n")