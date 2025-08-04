# parser.py
import fitz  # PyMuPDF
import tempfile
import requests
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
import os

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def parse_pdf(path: str, from_url: bool = True) -> Tuple[List[str], List[Dict]]:
    if from_url:
        response = requests.get(path)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            path = tmp.name

    doc = fitz.open(path)
    full_text, page_map = [], []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        full_text.append(text)
        page_map.append({"page": i + 1, "text": text})

    doc.close()
    if from_url:
        os.remove(path)

    return chunk_texts(page_map)

def chunk_texts(page_map: List[Dict]) -> Tuple[List[str], List[Dict]]:
    all_text = "\n".join(p["text"] for p in page_map)
    tokens = tokenizer.tokenize(all_text)

    chunk_size, overlap = 500, 100
    chunks, metadata = [], []

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        metadata.append({"start_token": i, "end_token": i + len(chunk_tokens)})
        i += chunk_size - overlap

    return chunks, metadata

def parse_pdf_from_url(url: str):
    return parse_pdf(url, from_url=True)

def parse_pdf_from_file(file_path: str):
    return parse_pdf(file_path, from_url=False)
