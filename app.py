import os
import json
import numpy as np
import streamlit as st

from src.rag_pipeline import answer_question_extractive


# ----------------------
# Streamlit page config
# ----------------------
st.set_page_config(
    page_title="General Health RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("General Health Patient Education Chatbot (RAG)")
st.write(
    "This chatbot answers questions based on patient information brochures "
    "stored in the app (PDF/TXT in the `data/brochures` folder). "
    "It uses a Retrieval-Augmented Generation (RAG) pipeline over a precomputed index. "
    "No internet search and no external LLM APIs are used."
)


# ----------------------
# Paths and session state
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BROCHURES_DIR = os.path.join(BASE_DIR, "data", "brochures")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
EMB_PATH = os.path.join(VECTOR_STORE_DIR, "embeddings.npy")
META_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json")

if "index_ready" not in st.session_state:
    st.session_state["index_ready"] = False
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of {question, answer, sources}


# ----------------------
# Load precomputed vector store
# ----------------------
def load_vector_store():
    if not os.path.exists(EMB_PATH) or not os.path.exists(META_PATH):
        st.error(
            "Vector store not found.\n\n"
            "Please run `build_vector_store.py` locally to precompute embeddings "
            "and commit the `vector_store/` folder to the repository."
        )
        return False

    try:
        embeddings = np.load(EMB_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return False

    st.session_state["embeddings"] = embeddings
    st.session_state["metadata"] = metadata
    st.session_state["index_ready"] = True
    return True


if not st.session_state["index_ready"]:
    load_vector_store()


# ----------------------
# Sidebar: list brochures only
# ----------------------
with st.sidebar:
    st.header("📂 Documents")

    if os.path.isdir(BROCHURES_DIR):
        files_in_dir = os.listdir(BROCHURES_DIR)
        brochure_files = [
            f for f in files_in_dir if f.lower().endswith((".pdf", ".txt"))
        ]
        if brochure_files:
            for f in brochure_files:
                st.write("• " + f)
        else:
            st.warning("No PDF/TXT files found in `data/brochures`.")
    else:
        st.error("Folder `data/brochures` does not exist.")


# ----------------------
# Main chat interface
# ----------------------
st.markdown("## 💬 Ask a health question")

if not st.session_state["index_ready"]:
    st.warning(
        "The document index is not ready.\n\n"
        "The vector store is missing or could not be loaded."
    )
else:
    # Show previous conversation
    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander("Sources", expanded=False):
                if not turn["sources"]:
                    st.write("No sources found.")
                else:
                    for i, src in enumerate(turn["sources"], start=1):
                        display_name = os.path.basename(src["filename"])
                        st.markdown(f"**Source {i}: {display_name}**")
                        st.write(f"Score: `{src['score']:.3f}`")
                        st.write(f"Chunk index: `{src['chunk_index']}`")
                        st.text(src["snippet"])
                        st.markdown("---")

    # New question
    user_question = st.chat_input("Type your question here...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and building answer..."):
                result = answer_question_extractive(
                    query=user_question,
                    doc_embeddings=st.session_state["embeddings"],
                    metadata=st.session_state["metadata"],
                    k=3,
                    max_chunk_chars=600,
                )

            st.write(result["answer"])

            with st.expander("Sources", expanded=False):
                if not result["sources"]:
                    st.write("No sources found.")
                else:
                    for i, src in enumerate(result["sources"], start=1):
                        display_name = os.path.basename(src["filename"])
                        st.markdown(f"**Source {i}: {display_name}**")
                        st.write(f"Score: `{src['score']:.3f}`")
                        st.write(f"Chunk index: `{src['chunk_index']}`")
                        st.text(src["snippet"])
                        st.markdown("---")

        # Save in history
        st.session_state["chat_history"].append(
            {
                "question": user_question,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        )

st.markdown("---")
st.caption(
    "This chatbot provides general educational information only "
    "and does not replace medical advice. For any concerns about your health, "
    "please consult a qualified healthcare professional."
)


