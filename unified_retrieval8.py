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
            "engagement": "Class Engagement & Attendance",
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
            "college_map": ["where", "location", "find", "map", "building", "room", "lab", "library", "canteen", "direction", "way", "office", "department", "block", "cse", "ece", "mechanical", "civil", "principal", "hod", "faculty"]
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
    
    def extract_filters(self, query: str) -> Dict[str, any]:
        """
        Extract filtering criteria from the query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with filter criteria (section, day, class, etc.)
        """
        query_lower = query.lower()
        filters = {}
        
        # Extract section (A, B, C, etc.)
        section_match = re.search(r'\b(section|class)\s+([a-z])\b', query_lower)
        if section_match:
            filters['section'] = section_match.group(2).upper()
        
        # Extract day of week
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day in query_lower:
                filters['day'] = day.capitalize()
                break
        
        return filters
    
    def detect_intent(self, query: str) -> Optional[str]:
        """
        Intelligently detect which dataset is most relevant to the query.
        
        Args:
            query: User query string
            
        Returns:
            Dataset name or None for global search
        """
        query_lower = query.lower()
        
        # PRIORITY RULES (check these first - most specific to least specific)
        
        # Engagement-related queries (CHECK FIRST - highest priority for attendance)
        # More aggressive detection for engagement dataset
        engagement_triggers = [
            "attendance rate" in query_lower,
            "attendance for" in query_lower,
            "participation rate" in query_lower,
            "engagement rate" in query_lower,
            "engagement metric" in query_lower,
            "participation score" in query_lower,
            "assignment rate" in query_lower,
            "assignment completion" in query_lower,
            ("show" in query_lower and "attendance" in query_lower),
            ("display" in query_lower and "attendance" in query_lower),
            ("show" in query_lower and "engagement" in query_lower),
            ("display" in query_lower and "engagement" in query_lower),
            ("show" in query_lower and "participation" in query_lower),
            ("display" in query_lower and "participation" in query_lower),
            "questions asked" in query_lower,
            "interaction count" in query_lower,
            "feedback rating" in query_lower
        ]
        
        if any(engagement_triggers):
            return "engagement"
        
        # Also check for "attendance" without schedule-related words
        if "attendance" in query_lower and not any(word in query_lower for word in ["schedule", "timetable", "when", "time"]):
            return "engagement"
        
        # Student-related queries
        if ("list" in query_lower or "show" in query_lower) and "student" in query_lower and "section" not in query_lower:
            return "students"
        
        # Event queries - IMPROVED (check earlier!)
        if any(word in query_lower for word in ["when is", "details about", "tell me about"]) and \
           any(word in query_lower for word in ["hackathon", "fest", "event", "workshop", "seminar", "competition", "cultural", "technical"]):
            return "events"
        
        # Schedule/Timetable queries
        if "schedule" in query_lower or "timetable" in query_lower:
            return "timetable"
        
        # Location queries
        if any(word in query_lower for word in ["where is", "find", "location of", "how to reach", "directions to"]):
            return "college_map"
        
        # FALLBACK: Keyword counting for ambiguous queries
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
    
    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, any], dataset_name: str) -> pd.DataFrame:
        """
        Apply filters to a dataframe based on extracted criteria.
        
        Args:
            df: DataFrame to filter
            filters: Dictionary of filter criteria
            dataset_name: Name of the dataset being filtered
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Apply section filter for timetable
        if dataset_name == "timetable" and "section" in filters:
            section = filters["section"]
            # Check if 'Section' column exists
            if 'Section' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Section'].str.upper() == section]
            elif 'Class' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Class'].str.upper() == section]
        
        # Apply day filter for timetable
        if dataset_name == "timetable" and "day" in filters:
            day = filters["day"]
            if 'Day' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Day'].str.lower() == day.lower()]
        
        # Apply section filter for students
        if dataset_name == "students" and "section" in filters:
            section = filters["section"]
            if 'Section' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Section'].str.upper() == section]
        
        # Apply class/section filter for engagement (special binary column structure)
        if dataset_name == "engagement" and "section" in filters:
            section = filters["section"]
            # Engagement dataset has binary indicator columns like "Class A", "Class B", etc.
            class_col_name = f"Class {section}"
            if class_col_name in filtered_df.columns:
                # Filter where the class column = 1 (student belongs to that class)
                filtered_df = filtered_df[filtered_df[class_col_name] == 1]
            # Fallback to standard column names
            elif 'Class' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Class'].str.upper() == section]
            elif 'Section' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Section'].str.upper() == section]
        
        return filtered_df
    
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
        threshold: float = 0.3,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search within a specific dataset with optional filtering."""
        if dataset_name not in self.datasets:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self.datasets.keys())}"
            )
        
        df = self.datasets[dataset_name]
        
        # Apply filters if provided
        if filters:
            df = self.apply_filters(df, filters, dataset_name)
            
            # If filtering resulted in empty dataframe, return empty results
            if len(df) == 0:
                return []
        
        # Get embeddings for the filtered indices
        if filters and len(df) < len(self.datasets[dataset_name]):
            # We have a filtered dataframe - need to get corresponding embeddings
            original_indices = df.index.tolist()
            embs = self.embeddings[dataset_name][original_indices]
        else:
            embs = self.embeddings[dataset_name]
        
        query_embedding = self.embed_query(query)
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
                    "index": df.index[idx]  # Use the dataframe's index
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
        
        if any(clubs_indicators):
            specific_queries = ['music club', 'coding club', 'drama club', 'dance club', 
                              'robotics club', 'photo', 'art club', 'sports club',
                              'when does', 'where does', 'who runs', 'coordinator']
            
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
            return [{
                "dataset": "quick_answer",
                "score": 1.0,
                "row": {"answer": quick_answer},
                "index": 0
            }]
        
        # Extract filters from query
        filters = self.extract_filters(query)
        
        # Detect most relevant dataset
        target_dataset = self.detect_intent(query)
        
        # DEBUG: Show what dataset was detected
        print(f"[DEBUG] Detected dataset: {target_dataset}")
        print(f"[DEBUG] Extracted filters: {filters}")
        
        # SPECIAL HANDLING for schedule queries - need more results with lower threshold
        if target_dataset == "timetable":
            results = self.search_dataset(query, target_dataset, top_k=10, threshold=0.15, filters=filters)
            if results:
                return results
        
        # SPECIAL HANDLING for engagement queries
        if target_dataset == "engagement":
            print(f"[DEBUG] Searching engagement dataset...")
            print(f"[DEBUG] Total engagement rows before filter: {len(self.datasets['engagement'])}")
            print(f"[DEBUG] Engagement embeddings shape: {self.embeddings['engagement'].shape}")
            
            # Apply filters manually to see what happens
            if filters and 'section' in filters:
                test_df = self.datasets['engagement'].copy()
                class_col = f"Class {filters['section']}"
                print(f"[DEBUG] Looking for column: {class_col}")
                if class_col in test_df.columns:
                    filtered = test_df[test_df[class_col] == 1]
                    print(f"[DEBUG] Rows after filtering: {len(filtered)}")
                else:
                    print(f"[DEBUG] Column {class_col} not found!")
            
            # Try with very low threshold to see if ANY matches exist
            results = self.search_dataset(query, target_dataset, top_k=5, threshold=0.01, filters=filters)
            print(f"[DEBUG] Engagement results found (threshold=0.01): {len(results)}")
            if results:
                print(f"[DEBUG] Top result score: {results[0]['score']:.4f}")
                return results
            else:
                print(f"[DEBUG] No results even at 0.01 threshold!")
                print(f"[DEBUG] Trying without filters...")
                results = self.search_dataset(query, target_dataset, top_k=5, threshold=0.01, filters=None)
                print(f"[DEBUG] Results without filters (threshold=0.01): {len(results)}")
                if results:
                    print(f"[DEBUG] Top result score: {results[0]['score']:.4f}")
                    return results
        
        # SPECIAL HANDLING for location queries - return only top 1 result
        if target_dataset == "college_map":
            results = self.search_dataset(query, target_dataset, top_k=1, threshold=0.25, filters=filters)
            if results:
                return results
        
        if target_dataset:
            # Search specific dataset with filters
            results = self.search_dataset(query, target_dataset, top_k=top_k, threshold=0.25, filters=filters)
            if results:
                return results
        
        # Fallback to global search
        print(f"[DEBUG] Falling back to global search")
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
            
            # Add section info if available
            if 'Section' in row and row['Section']:
                lines.append(f"Section: {row['Section']}")
            elif 'Class' in row and row['Class']:
                lines.append(f"Class: {row['Class']}")
            
            lines.append(f"Day: {row.get('Day', 'Not specified')}")
            lines.append(f"Time: {row.get('Time', 'Not specified')}")
            lines.append(f"Room: {row.get('Venue', 'Not specified')}")
            
            if 'Faculty' in row and row['Faculty']:
                lines.append(f"Instructor: {row['Faculty']}")
        
        elif result['dataset'] == 'college_map':
            # Primary location info
            building = row.get('Building_Name', 'Not specified')
            room = row.get('Room_Name', '')
            department = row.get('Department', '')
            
            # Build location string
            location_parts = []
            if room and room != '-':
                location_parts.append(room)
            if department and department != '-':
                location_parts.append(department)
            if building and building != '-':
                location_parts.append(building)
            
            location = ' - '.join(location_parts) if location_parts else 'Not specified'
            lines.append(f"Location: {location}")
            
            # Block/Building code
            if 'Block_Code' in row and row['Block_Code'] and row['Block_Code'] != '-':
                lines.append(f"Block: {row['Block_Code']}")
            
            # Floor
            if 'Floor_No' in row and str(row['Floor_No']) != 'Not Specified':
                floor_text = f"Floor {row['Floor_No']}" if row['Floor_No'] != 0 else "Ground Floor"
                lines.append(f"Floor: {floor_text}")
            
            # Faculty/Staff info
            if 'Faculty_Name' in row and row['Faculty_Name'] and row['Faculty_Name'] != '-':
                lines.append(f"Contact Person: {row['Faculty_Name']}")
                if 'Designation' in row and row['Designation'] and row['Designation'] != '-':
                    lines.append(f"Designation: {row['Designation']}")
            
            # Contact
            if 'Contact' in row and row['Contact'] and row['Contact'] != '-':
                lines.append(f"Email: {row['Contact']}")
            
            # Landmark/Directions
            if 'Landmark' in row and row['Landmark']:
                lines.append(f"Nearby: {row['Landmark']}")
            
            if 'Special_Guidance' in row and row['Special_Guidance']:
                lines.append(f"Note: {row['Special_Guidance']}")
            
            # Additional notes
            if 'Notes' in row and row['Notes']:
                lines.append(f"Timing: {row['Notes']}")
        
        elif result['dataset'] == 'engagement':
            # Determine which class this data belongs to
            class_name = None
            for col in ['Class A', 'Class B', 'Class C', 'Class D', 'Class E', 'Class F']:
                if col in row and row[col] == 1:
                    class_name = col.replace('Class ', '')
                    break
            
            if class_name:
                lines.append(f"Class: {class_name}")
            
            if 'Class_advisor' in row and row['Class_advisor']:
                lines.append(f"Class Advisor: {row['Class_advisor']}")
            
            if 'Week_number' in row:
                lines.append(f"Week: {row['Week_number']}")
            
            # Attendance information
            if 'Attendance_rate-scale' in row:
                attendance = row['Attendance_rate-scale']
                attendance_pct = f"{attendance * 100:.1f}%" if attendance <= 1 else f"{attendance:.1f}%"
                lines.append(f"Attendance Rate: {attendance_pct}")
            
            # Assignment completion
            if 'Assignment_rate-scale' in row:
                assignment = row['Assignment_rate-scale']
                assignment_pct = f"{assignment * 100:.1f}%" if assignment <= 1 else f"{assignment:.1f}%"
                lines.append(f"Assignment Completion: {assignment_pct}")
            
            # Participation metrics
            if 'Participation_score' in row:
                participation = row['Participation_score']
                participation_pct = f"{participation * 100:.1f}%" if participation <= 1 else f"{participation:.1f}%"
                lines.append(f"Overall Participation: {participation_pct}")
            
            if 'Math_parti_scale' in row:
                math_parti = row['Math_parti_scale']
                math_pct = f"{math_parti * 100:.1f}%" if math_parti <= 1 else f"{math_parti:.1f}%"
                lines.append(f"Math Participation: {math_pct}")
            
            if 'Sci_parti_scale' in row:
                sci_parti = row['Sci_parti_scale']
                sci_pct = f"{sci_parti * 100:.1f}%" if sci_parti <= 1 else f"{sci_parti:.1f}%"
                lines.append(f"Science Participation: {sci_pct}")
            
            # Engagement metrics
            if 'Questions_asked' in row:
                lines.append(f"Questions Asked: {row['Questions_asked']}")
            
            if 'Interaction_count' in row:
                lines.append(f"Interactions: {row['Interaction_count']}")
            
            if 'Feedback_rating scaled' in row:
                feedback = row['Feedback_rating scaled']
                feedback_pct = f"{feedback * 100:.1f}%" if feedback <= 1 else f"{feedback:.1f}%"
                lines.append(f"Feedback Rating: {feedback_pct}")
        
        else:
            # Generic display for students or other datasets
            # Show first 5 most important fields
            important_fields = ['Name', 'Title', 'Department', 'Email', 'Phone', 'Section', 'Class']
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
    print("  - Show Monday schedule for section A")
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
    print("  - Show attendance rate for class B")
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