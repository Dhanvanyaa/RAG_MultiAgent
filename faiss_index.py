
from sentence_transformers import SentenceTransformer
import faiss
import os


with open("chunks/nlp_chunks.txt", "r", encoding="utf-8") as f:
    raw_chunks = f.read().split("\n---\n")


model = SentenceTransformer("all-MiniLM-L6-v2")


embeddings = model.encode(raw_chunks)


dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


faiss.write_index(index, "embeddings/faiss_index.idx")
with open("embeddings/chunks.txt", "w", encoding="utf-8") as f:
    for chunk in raw_chunks:
        f.write(chunk + "\n---\n")
