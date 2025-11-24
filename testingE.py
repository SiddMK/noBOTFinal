import pandas as pd

# Load the engagement dataset
df = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SQL\\noBOT\\data\\Engagement_dataset_cleaned.csv")

print("="*60)
print("ENGAGEMENT DATASET DIAGNOSTICS")
print("="*60)
print(f"\nTotal rows: {len(df)}")
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nData types:")
print(df.dtypes)
print(f"\nSample values from each column:")
for col in df.columns:
    print(f"\n{col}: {df[col].dropna().head(2).tolist()}")