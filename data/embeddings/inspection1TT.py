import pickle

file_path = r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\embeddings\timetable_with_embeddings.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys())
