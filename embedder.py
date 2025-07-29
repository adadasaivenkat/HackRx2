import google.generativeai as genai
from typing import List
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = "models/embedding-001"

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for chunk in chunks:
        result = genai.embed_content(model=EMBED_MODEL, content=chunk, task_type="retrieval_document")
        embeddings.append(result["embedding"])
    return embeddings 