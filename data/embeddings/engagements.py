import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load engagement dataset
df = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\Engagement_dataset_cleaned.csv")

# Create meaningful text descriptions for each row
descriptions = []
for idx, row in df.iterrows():
    # Determine which class
    class_name = None
    for col in ['Class A', 'Class B', 'Class C', 'Class D', 'Class E', 'Class F']:
        if row[col] == 1:
            class_name = col.replace('Class ', '')
            break
    
    # Build rich text description
    desc = f"Class {class_name} Week {row['Week_number']} "
    desc += f"Advisor {row['Class_advisor']} "
    desc += f"Attendance rate {row['Attendance_rate-scale']:.2%} "
    desc += f"Assignment completion {row['Assignment_rate-scale']:.2%} "
    desc += f"Participation score {row['Participation_score']:.2%} "
    desc += f"Math participation {row['Math_parti_scale']:.2%} "
    desc += f"Science participation {row['Sci_parti_scale']:.2%} "
    desc += f"Questions asked {row['Questions_asked']} "
    desc += f"Interactions {row['Interaction_count']} "
    desc += f"Feedback rating {row['Feedback_rating scaled']:.2%} "
    desc += "engagement metrics attendance participation assignment questions interactions feedback class performance"
    
    descriptions.append(desc)

# Generate embeddings
print(f"Generating embeddings for {len(descriptions)} rows...")
embeddings = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)

# Save
output_path = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\embeddings\\engagement_embeddings.npy"
np.save(output_path, embeddings)
print(f"\n✅ Saved new embeddings to {output_path}")
print(f"Shape: {embeddings.shape}")
print("\nNow restart your noBOT system and try the queries again!")

## What this does:

#Each row will become text like:
#```
#"Class B Week 1 Advisor Tisca Chopra Attendance rate 5.13% Assignment completion 60.71% Participation score 57.11% Math participation 44.00% Science participation 24.32% Questions asked 0.0 Interactions 1 Feedback rating 50.00% engagement metrics attendance participation assignment questions interactions feedback class performance""""