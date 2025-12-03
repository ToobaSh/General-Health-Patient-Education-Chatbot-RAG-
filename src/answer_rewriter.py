from typing import List, Dict, Any, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


# Small local text-to-text model for rewriting answers
MODEL_NAME = "google/flan-t5-small"

_text_gen_pipe = None


def _get_text_gen_pipeline():
    """Load and cache the text2text-generation pipeline."""
    global _text_gen_pipe
    if _text_gen_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _text_gen_pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    return _text_gen_pipe


def rewrite_answer(
    question: str,
    extractive_answer: str,
    sources: List[Dict[str, Any]],
    language: str = "en",
    max_new_tokens: int = 128,
) -> Optional[str]:
    """Rewrite the extractive answer using a small local model.

    If something goes wrong, returns None so the app can fall back
    to the original answer.
    """
    if not extractive_answer.strip():
        return None

    try:
        pipe = _get_text_gen_pipeline()
    except Exception as e:
        print("Error loading local rewrite model:", e)
        return None

    instruction = (
        "Rewrite the answer below for a patient using clear, simple and reassuring language. "
        "Do not give personalized diagnosis. Do not add new medical facts. "
        "Finish with one sentence reminding that this does not replace advice "
        "from a healthcare professional.\n\n"
    )

    prompt = (
        instruction
        + "Question: " + question.strip() + "\n\n"
        + "Answer to rewrite:\n"
        + extractive_answer.strip()
    )

    try:
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        rewritten = outputs[0]["generated_text"].strip()
        return rewritten
    except Exception as e:
        print("Error during local rewriting:", e)
        return None
