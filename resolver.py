import os
from typing import List
import google.generativeai as genai
from embedder import embed_chunks
from retriever import retrieve_top_k

# Configure Gemini API
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
    for q in questions:
        try:
            # Embed question
            q_emb = embed_chunks([q])[0]

            # Retrieve top-k relevant chunks
            top_k_idx = retrieve_top_k(q_emb, index, k=3)
            context = "\n\n".join([chunks[i] for i in top_k_idx])

            # Create prompt
            prompt = f"""
You are an expert legal/compliance assistant. Based on the following document context, answer the question clearly and accurately.

Document Context:
{context}

Question: {q}
"""

            # Generate answer
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Clean answer prefix
            for prefix in ["**Answer:**", "Answer:", "**Answer**:", "**Answer**", "answer:", "answer"]:
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip(" :.-")
            answers.append(text)

        except Exception as e:
            answers.append(f"Could not process question '{q}': {str(e)}")

    return answers
