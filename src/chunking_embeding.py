import os
import pandas as pd
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# === Step 1: Load cleaned and filtered data ===
data_path = "../data/filtered_complaints.csv"
df = pd.read_csv(data_path)

if 'cleaned_narrative' not in df.columns:
    raise ValueError("❌ Missing 'cleaned_narrative' column. Please run text cleaning first.")

print(f"✅ Loaded {len(df)} cleaned complaints.")

# === Step 2: Chunking using LangChain RecursiveCharacterTextSplitter ===
chunk_size = 300
chunk_overlap = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", " "]
)

documents = []
metadatas = []

for idx, row in df.iterrows():
    chunks = splitter.split_text(row['cleaned_narrative'])
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "complaint_id": row['Complaint ID'],
            "product": row['Product'],
            "chunk_id": i
        })

print(f"✅ Created {len(documents)} chunks from {len(df)} complaints.")

# === Step 3: Generate embeddings using sentence-transformers ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, show_progress_bar=True)

print(f"✅ Embeddings shape: {embeddings.shape}")

# === Step 4: Index the embeddings using FAISS ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === Step 5: Save FAISS index and metadata ===
vector_dir = "../vector_store/faiss_index"
os.makedirs(vector_dir, exist_ok=True)

faiss.write_index(index, os.path.join(vector_dir, "complaints.index"))

with open(os.path.join(vector_dir, "metadata.pkl"), "wb") as f:
    pickle.dump(metadatas, f)

print(f"✅ Saved FAISS index and metadata to: {vector_dir}")
