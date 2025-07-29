import fitz  # PyMuPDF
import tempfile
import requests
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
import os

def parse_pdf_from_url(url: str) -> Tuple[List[str], List[Dict]]:
    # Download PDF to temp file
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    # Parse PDF
    doc = fitz.open(tmp_path)
    full_text = []
    page_map = []
    for i, page in enumerate(doc):
        text = page.get_text()
        full_text.append(text)
        page_map.append({"page": i+1, "text": text})
    doc.close()
    os.remove(tmp_path)
    # Tokenize and chunk
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    all_text = "\n".join([p["text"] for p in page_map])
    tokens = tokenizer.tokenize(all_text)
    chunk_size = 500
    overlap = 50
    chunks = []
    chunk_metadata = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        chunk_metadata.append({"start_token": i, "end_token": i+len(chunk_tokens)})
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks, chunk_metadata

def parse_pdf_from_file(file_path: str) -> Tuple[List[str], List[Dict]]:
    # Parse PDF from local file
    doc = fitz.open(file_path)
    full_text = []
    page_map = []
    for i, page in enumerate(doc):
        text = page.get_text()
        full_text.append(text)
        page_map.append({"page": i+1, "text": text})
    doc.close()
    # Tokenize and chunk
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    all_text = "\n".join([p["text"] for p in page_map])
    tokens = tokenizer.tokenize(all_text)
    chunk_size = 500
    overlap = 50
    chunks = []
    chunk_metadata = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        chunk_metadata.append({"start_token": i, "end_token": i+len(chunk_tokens)})
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks, chunk_metadata 