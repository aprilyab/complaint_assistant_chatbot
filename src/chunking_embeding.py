import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import numpy as np
from uuid import uuid4

# === ✅ Parameters ===
DATA_PATH = r"C:\Users\user\Desktop\tasks\complaint_assistant_chatbot\data\raw\filtered_complaints.csv"
VECTOR_DIR = r"C:\Users\user\Desktop\tasks\complaint_assistant_chatbot\vector_store"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# === ✅ Ensure directory exists ===
os.makedirs(VECTOR_DIR, exist_ok=True)

# === ✅ Load dataset ===
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} complaints with narratives.")
except FileNotFoundError:
    print(f"❌ File not found: {DATA_PATH}")
    exit(1)

# === ✅ Chunking function ===
def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# === ✅ Create text chunks with metadata ===
chunk_records = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Chunking text"):
    text = str(row.get("cleaned_narrative", "")).strip()
    if not text:
        continue
    chunks = chunk_text(text)
    for chunk in chunks:
        chunk_records.append({
            "id": str(uuid4()),
            "text": chunk,
            "product": row.get("Product", "")
        })

print(f"✅ Generated {len(chunk_records)} chunks.")

# === ✅ Load embedding model ===
print("🔍 Loading embedding model (all-MiniLM-L6-v2)...")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# === ✅ Embed chunks ===
texts = [rec["text"] for rec in chunk_records]
print(f"🔍 Generating embeddings for {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

# === ✅ FAISS Index creation ===
embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings).astype("float32"))
print("✅ FAISS index built and populated.")

# === ✅ Save index and metadata ===
index_path = os.path.join(VECTOR_DIR, "faiss_index.bin")
meta_path = os.path.join(VECTOR_DIR, "metadata.csv")

faiss.write_index(index, index_path)
pd.DataFrame(chunk_records).to_csv(meta_path, index=False)

# === ✅ Verify output files ===
if os.path.exists(index_path) and os.path.exists(meta_path):
    print(f"✅ Vector store saved to: {VECTOR_DIR}")
    print(f"  - faiss_index.bin: {os.path.getsize(index_path)} bytes")
    print(f"  - metadata.csv: {os.path.getsize(meta_path)} bytes")
else:
    print("❌ Failed to save one or both output files.")
