
from typing import List, Dict, Any
import textwrap
import numpy as np
from .retriever import retrieve_top_k
import os

def _clean_text(text: str) -> str:
    """Remove weird line breaks and extra spaces."""
    text = text.replace("\r", " ").replace("\n", " ")
    while "  " in text:
        text = text.replace("  " , " ")
    return text.strip()


def _keep_first_sentences(text: str, max_sentences: int = 3) -> str:
    """Keep only the first N sentences to avoid long, messy blocks."""
    text = _clean_text(text)
    # Very simple sentence split
    parts = text.split(". ")
    if len(parts) <= max_sentences:
        return text
    kept = ". ".join(parts[:max_sentences])
    # Add a dot if missing at the end
    kept = kept.strip()
    if not kept.endswith("."):
        kept += "."
    return kept


def answer_question_extractive(
    query: str,
    doc_embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    k: int = 3,
    max_chunk_chars: int = 600,
) -> Dict[str, Any]:
    """
    Extractive RAG-style answer:
    - retrieve top-k chunks
    - build a readable explanation from them
    - NO LLM, only basic text cleaning / formatting

    Returns:
        {
            "question": str,
            "answer": str,
            "sources": [
                {
                    "filename": str,
                    "score": float,
                    "chunk_index": int,
                    "snippet": str,
                },
                ...
            ]
        }
    """
    results = retrieve_top_k(
        query=query,
        doc_embeddings=doc_embeddings,
        metadata=metadata,
        k=k,
    )

    if not results:
        return {
            "question": query,
            "answer": (
                "I could not find any relevant information about this in the loaded documents. "
                "Please check that the PDFs contain information about this topic."
            ),
            "sources": [],
        }

    bullet_points = []
    sources_list = []

    for r in results:
        raw_text = r["text"][:max_chunk_chars]
        short_snippet = _keep_first_sentences(raw_text, max_sentences=3)

        filename = os.path.basename(r.get("filename", "unknown document"))

        bullet_points.append(f"- From **{filename}**: {short_snippet}")

        sources_list.append(
            {
                "filename": filename,
                "score": r["score"],
                "chunk_index": r["chunk_index"],
                "snippet": short_snippet,
            }
        )

    explanation_lines = [
        f"Here is what the documents say about your question:",
        "",
    ]
    explanation_lines.extend(bullet_points)
    explanation_lines.append("")
    explanation_lines.append(
        "This summary is built directly from the brochures. "
        "It is general information and does **not** replace the opinion of a healthcare professional."
    )

    answer_text = "\n".join(explanation_lines)

    return {
        "question": query,
        "answer": answer_text,
        "sources": sources_list,
    }
