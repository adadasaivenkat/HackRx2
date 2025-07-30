from dotenv import load_dotenv
load_dotenv()
import os
print('GOOGLE_API_KEY loaded:', bool(os.getenv('GOOGLE_API_KEY')))

from fastapi import FastAPI, Request, HTTPException, status, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from parser import parse_pdf_from_url
from embedder import embed_chunks
from retriever import build_faiss_index, retrieve_top_k
from resolver import answer_questions_with_gemini

AUTH_TOKEN = "37e8fe4ec2f129635f9d5776a0696b0c811a9c48d4af221f470e3ead04aa1ca8"

app = FastAPI()

def clean_answer(text):
    for prefix in ["**Answer:**", "Answer:", "**Answer**:", "**Answer**", "answer:", "answer"]:
        if text.strip().lower().startswith(prefix.lower()):
            return text.strip()[len(prefix):].strip(" :.-")
    return text.strip()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/api/v1/hackrx/run")
async def run_query(request: Request, body: QueryRequest):
    print("Received request")
    auth = request.headers.get("authorization")
    if not auth or auth != f"Bearer {AUTH_TOKEN}":
        print("Authorization failed")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing authorization token.")

    print("Parsing PDF")
    chunks, chunk_metadata = parse_pdf_from_url(body.documents)
    print(f"Parsed PDF into {len(chunks)} chunks")

    print("Embedding chunks")
    embeddings = embed_chunks(chunks)
    print("Embeddings generated")

    print("Building FAISS index")
    index = build_faiss_index(embeddings)
    print("FAISS index built")

    print("Answering questions")
    answers = answer_questions_with_gemini(body.questions, chunks, chunk_metadata, embeddings, index)
    print("Questions answered")

    print("Returning response")
    simple_answers = [clean_answer(a) for a in answers]
    return JSONResponse({"answers": simple_answers})

@app.post("/api/v1/hackrx/upload")
async def run_query_upload(
    request: Request,
    file: UploadFile = File(...),
    questions: List[str] = Form(...)
):
    print("Received upload request")
    auth = request.headers.get("authorization")
    if not auth or auth != f"Bearer {AUTH_TOKEN}":
        print("Authorization failed (upload)")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing authorization token.")

    import tempfile
    import shutil
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with temp_file as f:
        shutil.copyfileobj(file.file, f)
    temp_path = temp_file.name
    print(f"Saved uploaded file to {temp_path}")

    from parser import parse_pdf_from_file
    print("Parsing uploaded PDF")
    chunks, chunk_metadata = parse_pdf_from_file(temp_path)
    print(f"Parsed PDF into {len(chunks)} chunks (upload)")

    print("Embedding chunks (upload)")
    embeddings = embed_chunks(chunks)
    print("Embeddings generated (upload)")

    print("Building FAISS index (upload)")
    index = build_faiss_index(embeddings)
    print("FAISS index built (upload)")

    print("Answering questions (upload)")
    answers = answer_questions_with_gemini(questions, chunks, chunk_metadata, embeddings, index)
    print("Questions answered (upload)")

    import os
    os.remove(temp_path)
    print(f"Removed temp file {temp_path}")

    print("Returning response (upload)")
    simple_answers = [clean_answer(a) for a in answers]
    return JSONResponse({"answers": simple_answers})

@app.get("/")
async def status():
    return JSONResponse({"status": "running"}) 
