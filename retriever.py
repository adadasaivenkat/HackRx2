# retriever.py
import faiss
import numpy as np
from typing import List

def build_faiss_index(embeddings: List[List[float]]):
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    xb = np.array(embeddings).astype('float32')
    faiss.normalize_L2(xb)
    index.add(xb)
    return index

def retrieve_top_k(query_emb: List[float], index, k=5) -> List[int]:
    xq = np.array([query_emb]).astype('float32')
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k)
    return I[0].tolist()
