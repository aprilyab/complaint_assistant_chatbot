import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Tuple

from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


class RAGPipeline:
    def __init__(self, vector_dir: str, embedding_model_name="all-MiniLM-L6-v2", generator_model_name="facebook/bart-large-cnn"):
        self.vector_dir = vector_dir
        
        # Load embedding model (same as for indexing)
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Load FAISS index
        index_path = os.path.join(vector_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        meta_path = os.path.join(vector_dir, "metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        self.metadata = pd.read_csv(meta_path)
        
        # Load generator LLM pipeline
        self.generator = pipeline("text2text-generation", model=generator_model_name)
        
        print(f" Loaded RAG pipeline with embedding model '{embedding_model_name}' and generator model '{generator_model_name}'")

    def retrieve(self, query: str, top_k=5) -> List[Tuple[str, float]]:
        """
        Retrieve top_k relevant chunks for the query.
        Returns list of (chunk_text, similarity_score).
        """
        query_emb = self.embedder.encode([query])
        D, I = self.index.search(np.array(query_emb).astype('float32'), top_k)  # D: distances, I: indices
        
        results = []
        for idx, dist in zip(I[0], D[0]):
            chunk_text = self.metadata.iloc[idx]["text"]
            similarity = 1 / (1 + dist)  # Convert L2 distance to similarity-like score
            results.append((chunk_text, similarity))
        return results

    def generate_prompt(self, context_chunks: List[str], question: str) -> str:
        """
        Construct the prompt for the LLM with retrieved context.
        """
        context_text = "\n---\n".join(context_chunks)
        prompt = (
            f"You are a financial analyst assistant for CrediTrust. "
            f"Your task is to answer questions about customer complaints. "
            f"Use the following retrieved complaint excerpts to formulate your answer. "
            f"If the context doesn't contain the answer, state that you don't have enough information.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return prompt

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer from the LLM given the prompt.
        """
        response = self.generator(prompt, max_length=256, do_sample=False)
        answer = response[0]['generated_text'].strip()
        return answer

    def answer_question(self, question: str, top_k=5) -> Tuple[str, List[str]]:
        """
        Complete pipeline: retrieve context chunks, generate prompt, and return answer.
        Returns (answer, retrieved_chunks).
        """
        retrieved = self.retrieve(question, top_k=top_k)
        chunks = [text for text, _ in retrieved]
        prompt = self.generate_prompt(chunks, question)
        answer = self.generate_answer(prompt)
        return answer, chunks

if __name__ == "__main__":
    # Example usage
    VECTOR_DIR = r"C:\Users\user\Desktop\tasks\complaint_assistant_chatbot\vector_store"
    rag = RAGPipeline(VECTOR_DIR)
    
    sample_questions = [
        "Why are customers complaining about the loan approval process?",
        "What issues do customers report regarding their credit cards?",
        "How does CrediTrust handle disputes in billing?",
        "Are there any common complaints about mobile banking?",
        "What is the most frequent complaint about customer service?"
    ]
    
    for q in sample_questions:
        print(f"\n Question: {q}")
        answer, sources = rag.answer_question(q, top_k=5)
        print(f" Answer: {answer}")
        print(f" Retrieved Sources (1-2): {sources[:2]}")
 
