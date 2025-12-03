import os
import streamlit as st

from src.ingestion import extract_texts_from_files
from src.chunker import chunk_documents
from src.embeddings import build_embeddings_from_chunks
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
    "This chatbot answers questions based on official health brochures "
    "stored locally in the app (PDF/TXT in the `data/brochures` folder). "
    "It uses a Retrieval-Augmented Generation (RAG) pipeline to search the documents "
    "and build an answer. It does not search the internet."
)


# ----------------------
# Paths and state init
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BROCHURES_DIR = os.path.join(BASE_DIR, "data", "brochures")

if "index_ready" not in st.session_state:
    st.session_state["index_ready"] = False
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "chat_history" not in st.session_state:
    # Each item: {"question": str, "answer": str, "sources": list}
    st.session_state["chat_history"] = []


def build_index() -> bool:
    """
    Build the RAG index:
      - read PDFs/TXTs from data/brochures
      - extract text
      - chunk documents
      - compute embeddings

    Returns True on success, False on failure.
    """
    if not os.path.isdir(BROCHURES_DIR):
        st.error(
            "The folder `data/brochures` does not exist.\n\n"
            "Create it next to `app.py` and add some PDF/TXT patient leaflets."
        )
        return False

    files_in_dir = os.listdir(BROCHURES_DIR)
    brochure_files = [
        f for f in files_in_dir if f.lower().endswith((".pdf", ".txt"))
    ]

    if not brochure_files:
        st.error(
            "No PDF or TXT files found in `data/brochures`.\n\n"
            "Please add some patient brochures (for example: diabetes, hypertension, MRI...)."
        )
        return False

    file_objects = []
    for fname in brochure_files:
        path = os.path.join(BROCHURES_DIR, fname)
        try:
            file_objects.append(open(path, "rb"))
        except Exception as e:
            st.error(f"Could not open {path}: {e}")

    if not file_objects:
        st.error("Could not open any documents. Check file permissions or paths.")
        return False

    # 1) Extract text
    with st.spinner("Extracting text from documents..."):
        texts_by_file = extract_texts_from_files(file_objects)

    # 2) Chunk documents
    with st.spinner("Splitting documents into chunks..."):
        chunks_by_file = chunk_documents(
            texts_by_file,
            chunk_size=800,
            chunk_overlap=200,
        )

    # 3) Build embeddings
    with st.spinner("Computing embeddings..."):
        embeddings, metadata = build_embeddings_from_chunks(chunks_by_file)

    # 👉 Clean filenames in metadata so we only keep the brochure name, not the full path
    for m in metadata:
        if "filename" in m and m["filename"]:
            m["filename"] = os.path.basename(m["filename"])

    st.session_state["embeddings"] = embeddings
    st.session_state["metadata"] = metadata
    st.session_state["index_ready"] = True

    
    return True


# Automatically build index on first load (if not ready)
if not st.session_state["index_ready"]:
    if os.path.isdir(BROCHURES_DIR):
        build_index()


# ----------------------
# Sidebar: show documents and status
# ----------------------
# ----------------------
# Sidebar: ONLY brochure names
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
        "The document index is not ready yet.\n\n"
        "Please add brochures to `data/brochures` and rebuild the index from the sidebar."
    )
else:
    # Show previous Q&A
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
                        # Show only the filename (already cleaned in metadata)
                        display_name = os.path.basename(src["filename"])
                        st.markdown(f"**Source {i}: {display_name}**")
                        st.write(f"Score: `{src['score']:.3f}`")
                        st.write(f"Chunk index: `{src['chunk_index']}`")
                        st.text(src["snippet"])
                        st.markdown("---")

    # New question input
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

        # Save in chat history
        st.session_state["chat_history"].append(
            {
                "question": user_question,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        )

st.markdown("---")
st.caption(
    "This chatbot provides general educational information only and does not replace medical advice. "
    "For any concerns about your health, please consult a qualified healthcare professional."
)
