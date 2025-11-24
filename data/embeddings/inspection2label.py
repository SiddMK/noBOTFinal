import pickle

file_path = r"C:\Users\ADMIN\OneDrive\Desktop\SQL\noBOT\data\embeddings\label_encoder.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys())
