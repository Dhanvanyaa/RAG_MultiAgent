from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import faiss
import torch
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity


sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token


def route_query(query):
    query_lower = query.lower()
    
    math_keywords = ["calculate", "solve", "derivative", "integrate", "limit", "equation"]
    if any(word in query_lower for word in math_keywords):
        return "math"
    
    if query_lower.startswith("define") or "meaning of" in query_lower or "definition of" in query_lower:
        return "dictionary"

    return "rag"


def handle_math(query):
    try:
        result = eval(query.split("calculate")[-1])
        return f"\n Math Result: {result}"
    except Exception:
        return "⚠️ Sorry, couldn't calculate that."


def handle_dictionary(query):
    word = query.lower().replace("define", "").replace("what is the meaning of", "").replace("definition of", "").strip()
    synsets = wordnet.synsets(word)
    if not synsets:
        return f" No definition found for '{word}'."
    
    definition = synsets[0].definition()
    examples = synsets[0].examples()
    response = f" Definition of '{word}':\n{definition}"
    if examples:
        response += f"\n\n Example: {examples[0]}"
    return response


def load_faiss_index(index_path):
    return faiss.read_index(index_path)


def load_chunks(chunks_file_path):
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        return f.read().split("\n---\n")


def retrieve_top_chunks(query, k=3, index=None, chunks=None):
    query_vec = sentence_model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]


def refine_answer(query, chunks):
    print("\n Top 3 Retrieved Chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{chunk.strip()}")

    all_sentences = []
    for chunk in chunks[:3]:
        sentences = [s.strip() for s in chunk.split('.') if s.strip()]
        all_sentences.extend(sentences)

    sentence_embeddings = sentence_model.encode(all_sentences)
    query_embedding = sentence_model.encode([query])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

    top_indices = similarities.argsort()[-3:][::-1]
    key_sentences = [all_sentences[i] for i in top_indices]

    context = '\n'.join(key_sentences)
    prompt = (
        "Based on this context:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Provide a concise answer:"
    )

    inputs = gpt2_tokenizer(prompt, return_tensors='pt', max_length=900, truncation=True)

    try:
        outputs = gpt2_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=80,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
        answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_answer = answer.split('Question:')[-1].split('Answer:')[-1].strip()
        return f"\n Generated Answer:\n{final_answer}"
    except Exception as e:
        return f"\n Answer generation failed. Most relevant context:\n{context}"


def main():
    faiss_index_path = "embeddings/faiss_index.idx"
    chunks_file_path = "embeddings/chunks.txt"

    index = load_faiss_index(faiss_index_path)
    chunks = load_chunks(chunks_file_path)

    query = input("🔹 Enter your question: ")

    route = route_query(query)

    print(f"\n Routing Decision: Using the **'{route}'** module.")

    if route == "math":
        print("\n Math Module Response:")
        print(handle_math(query))
    elif route == "dictionary":
        print("\n Dictionary Module Response:")
        print(handle_dictionary(query))
    else:
        print("\n RAG Pipeline Triggered...")
        retrieved_chunks = retrieve_top_chunks(query, k=3, index=index, chunks=chunks)
        print("\n Retrieved Context Snippets Used for Answer:")
        answer = refine_answer(query, retrieved_chunks)
        print("\n Final Answer:")
        print(answer)

if __name__ == "__main__":
    main()
