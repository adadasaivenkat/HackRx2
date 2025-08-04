from dotenv import load_dotenv
from fastapi import status
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import tempfile
import shutil 
load_dotenv()
import os
import tempfile
import shutil

from fastapi import FastAPI, Request, HTTPException, status, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from parser import parse_pdf_from_url, parse_pdf_from_file
from embedder import embed_chunks
from retriever import build_faiss_index
from resolver import answer_questions_with_gemini

AUTH_TOKEN = "37e8fe4ec2f129635f9d5776a0696b0c811a9c48d4af221f470e3ead04aa1ca8"

app = FastAPI()


def clean_answer(text):
    prefixes = ["**Answer:**", "Answer:", "**Answer**:", "**Answer**", "answer:", "answer"]
    for prefix in prefixes:
        if text.strip().lower().startswith(prefix.lower()):
            return text.strip()[len(prefix):].strip(" :.-")
    return text.strip()


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


def authorize_request(request: Request):
    auth = request.headers.get("authorization")
    if not auth or auth != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing authorization token.")


@app.post("/api/v1/hackrx/run")
async def run_query(request: Request, body: QueryRequest):
    print("Received URL-based query request")
    authorize_request(request)

    chunks, metadata = parse_pdf_from_url(body.documents)
    print(f"Parsed PDF into {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)

    answers = answer_questions_with_gemini(body.questions, chunks, metadata, embeddings, index)
    return JSONResponse({"answers": [clean_answer(a) for a in answers]})


@app.post("/api/v1/hackrx/upload")
async def run_query_upload(
    request: Request,
    file: UploadFile = File(...),
    questions: List[str] = Form(...)
):
    print("Received file upload request")
    authorize_request(request)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        file_path = tmp.name
    print(f"Uploaded file saved to {file_path}")

    chunks, metadata = parse_pdf_from_file(file_path)
    print(f"Parsed uploaded PDF into {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)

    answers = answer_questions_with_gemini(questions, chunks, metadata, embeddings, index)

    os.remove(file_path)
    print(f"Temporary file {file_path} removed")

    return JSONResponse({"answers": [clean_answer(a) for a in answers]})


@app.get("/")
async def status():
    return JSONResponse({"status": "running"})
