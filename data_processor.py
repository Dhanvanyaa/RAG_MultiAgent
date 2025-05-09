
import os
import fitz  

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


if __name__ == "__main__":
    text = pdf_to_text("data/nlp_intro.pdf") 
    chunks = chunk_text(text)

    
    with open("chunks/nlp_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")
