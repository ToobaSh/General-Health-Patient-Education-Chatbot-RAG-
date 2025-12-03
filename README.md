# General-Health-Patient-Education-Chatbot-RAG-

This is a simple medical chatbot that answers questions using only the information contained in local patient brochures (PDF/TXT files).  
The chatbot does not search the internet and does not use any paid LLM APIs.

It uses a RAG (Retrieval-Augmented Generation) pipeline:

- extract text from brochures  
- split text into chunks  
- create embeddings with sentence-transformers  
- retrieve the most relevant chunks  
- build an extractive answer with sources  

All answers come directly from the documents stored in `data/brochures`.

---
