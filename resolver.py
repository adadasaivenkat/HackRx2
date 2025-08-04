# resolver.py
import google.generativeai as genai
from typing import List
from embedder import embed_chunks
from retriever import retrieve_top_k
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
LLM_MODEL = "models/gemini-1.5-flash-latest"
model = genai.GenerativeModel(model_name=LLM_MODEL)

def answer_questions_with_gemini(
    questions: List[str],
    chunks: List[str],
    chunk_metadata: List[dict],
    chunk_embeddings: List[List[float]],
    index
) -> List[str]:
    answers = []
    for question in questions:
        try:
            q_emb = embed_chunks([question])[0]
            top_k_idx = retrieve_top_k(q_emb, index, k=5)
            context = "\n\n".join([chunks[i] for i in top_k_idx])

            prompt = f"""
You are a helpful assistant. Based on the document context, answer the question briefly and accurately.

Context:
{context}

Question: {question}
Answer:"""

            response = model.generate_content(prompt)
            answer = response.text.strip()
            for prefix in ["**Answer:**", "Answer:", "**Answer**", "answer:"]:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip(" :.-")
            answers.append(answer)

        except Exception as e:
            answers.append(f"Error answering: {question} — {str(e)}")

    return answers
