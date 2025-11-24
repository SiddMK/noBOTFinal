import pandas as pd

df = pd.read_csv("dataset_roadmap_cleaned.csv")

def row_to_text(row):
    return (
        f"{row['Faculty_Name']}, {row['Designation']} at {row['Department']} department, "
        f"sits in {row['Room_Name']} ({row['Room_Type']}) located in {row['Building_Name']} "
        f"Block {row['Block_Code']}, Floor {row['Floor_No']}. "
        f"Landmark: {row['Landmark']}. Notes: {row['Notes']}. "
        f"Special Guidance: {row['Special_Guidance']}."
    )

df["text_chunk"] = df.apply(row_to_text, axis=1)
df.to_csv("roadmap_rag_text.csv", index=False)
