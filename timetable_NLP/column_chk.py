import pandas as pd

# === DIAGNOSTIC CODE - Run this first ===
file_path = "timetable_dataset_2nd draft.xlsx"

# Load the Excel file
df = pd.read_excel(file_path)

print("=== EXCEL FILE DIAGNOSTICS ===\n")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}\n")

print("=== COLUMN NAMES ===")
print(df.columns.tolist())
print()

print("=== FIRST 5 ROWS OF DATA ===")
print(df.head())
print()

print("=== DATA TYPES ===")
print(df.dtypes)