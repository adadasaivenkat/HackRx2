# embedder.py
import google.generativeai as genai
from typing import List
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "models/embedding-001"

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for chunk in chunks:
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=chunk,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        except Exception as e:
            print(f"Embedding failed for chunk: {chunk[:50]}... Error: {e}")
            embeddings.append([0.0] * 768)  # fallback zero vector
    return embeddings
