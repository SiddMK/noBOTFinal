import pandas as pd

# === STEP 1: Load your Excel ===
file_path = "timetable_dataset_2nd draft.xlsx"
df = pd.read_excel(file_path)

# === STEP 2: Clean up the data ===
# Strip whitespace from all string columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()

# === STEP 3: Define a function to turn each row into a sentence ===
def make_sentence(row):
    # Get values using EXACT column names from your Excel
    section = row.get('Section', '')
    day = row.get('Day', '')
    period = row.get('Period_No', '')
    time = row.get('Time', '')
    subject = row.get('Subject', '')
    faculty = row.get('Faculty', '')
    subject_code = row.get('Subject Code', '')  # Note: space and capitals!
    room_no = row.get('Room_No', '')
    
    # Convert NaN to empty string
    section = str(section).strip() if pd.notna(section) else ''
    day = str(day).strip() if pd.notna(day) else ''
    period = str(period).strip() if pd.notna(period) else ''
    time = str(time).strip() if pd.notna(time) else ''
    subject = str(subject).strip() if pd.notna(subject) else ''
    faculty = str(faculty).strip() if pd.notna(faculty) else ''
    subject_code = str(subject_code).strip() if pd.notna(subject_code) else ''
    room_no = str(room_no).strip() if pd.notna(room_no) else ''
    
    # Fix encoding issues in time
    time = time.replace('â€"', '-').replace('–', '-')
    
    # Remove .0 from period number
    if period and period != 'nan':
        try:
            period = str(int(float(period)))
        except:
            pass
    
    # Handle break periods differently
    if subject and 'BREAK' in subject.upper():
        return f"{subject} on {day}" if day else subject
    
    # Skip rows with missing critical data
    if not section or not day or not subject or section == 'nan' or day == 'nan':
        return ''
    
    # Create the sentence
    return (
        f"On {day}, Section {section} has {subject} "
        f"(subject code {subject_code}) during period {period} at {time}, "
        f"taught by {faculty} in room {room_no}."
    )

# === STEP 4: Apply function to all rows ===
df['Sentence'] = df.apply(make_sentence, axis=1)

# === STEP 5: Save the result ===
output_path = "timetable_sentences_fixed.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"✓ Done! Sentences saved to: {output_path}")
print(f"✓ Total rows processed: {len(df)}")
print(f"✓ Non-empty sentences: {df['Sentence'].astype(bool).sum()}")
print("\n=== Preview of first 10 sentences ===")

count = 0
for i, sentence in enumerate(df['Sentence']):
    if sentence and count < 10:
        count += 1
        print(f"{count}. {sentence}")
