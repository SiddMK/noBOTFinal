# debug_datasets.py
# Run this to check what's in your datasets

import pandas as pd
from pathlib import Path

base_path = Path("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT")
data_dir = base_path / "data"

print("="*60)
print("DATASET INSPECTION TOOL")
print("="*60)

datasets = {
    "students": "noBOT_dataset_cleaned.csv",
    "engagement": "Engagement_dataset_cleaned.csv",
    "timetable": "timetable_dataset.csv",
    "events": "dataset_events1.csv",
    "college_map": "dataset_roadmap_cleaned.csv"
}

for name, filename in datasets.items():
    print(f"\n{'='*60}")
    print(f"📊 DATASET: {name.upper()}")
    print('='*60)
    
    try:
        if name == "timetable":
            df = pd.read_csv(data_dir / filename, encoding="latin1", engine="python")
        else:
            df = pd.read_csv(data_dir / filename)
        
        print(f"\n✅ Loaded successfully")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\n📋 First 3 rows:")
        print("-"*60)
        print(df.head(3).to_string())
        
        print(f"\n🔍 Sample data from first row:")
        print("-"*60)
        for col in df.columns:
            value = df.iloc[0][col]
            print(f"  {col}: {value}")
        
    except Exception as e:
        print(f"\n❌ Error loading {name}: {e}")

print("\n" + "="*60)
print("INSPECTION COMPLETE")
print("="*60)