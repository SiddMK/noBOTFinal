import chromadb

# Initialize once (don’t reload each time)
client = chromadb.Client()
collection = client.get_or_create_collection("timetable")

def get_timetable_answer(query, n_results=3):
    """Retrieve best timetable info based on user query"""
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results['documents'][0]
    return docs
