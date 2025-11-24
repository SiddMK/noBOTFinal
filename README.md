# 🎓 noBOT – Smart AI Campus Assistant  
**An AI-powered student assistant built using RAG + semantic search**

noBOT is an intelligent campus assistant designed to answer student queries related to:  
- Student information  
- Timetables  
- Teacher interactions  
- Campus map  
- Upcoming events  

It uses a **unified retrieval system** + **embeddings** + **RAG pipeline** to deliver accurate and context-aware responses.

---

## 🚀 Features

### 🔍 **1. Unified Semantic Retrieval**
- Searches across multiple datasets (Students, Timetable, Engagement, Events, Campus Map)
- Powered by `SentenceTransformer` embeddings
- Uses cosine similarity to fetch the best match

### 🧠 **2. RAG Pipeline**
- The retriever fetches the most relevant context  
- LLM generates accurate answers using that context  
- Avoids hallucination and improves accuracy

### 🗂️ **3. Modular Dataset Design**
Datasets planned & implemented:
- ✔ Student Information  
- ✔ Student–Teacher Engagement  
- ✔ Timetable (with RAG integrated)  
- ⏳ Campus Map Dataset  
- ⏳ Events Dataset  

### 💻 **4. Streamlit Frontend**
- Clean and simple UI  
- Instant Q&A experience  
- Ready for deployment  

---

## 🏗️ Project Architecture

