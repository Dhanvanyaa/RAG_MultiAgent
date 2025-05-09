import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import faiss
import torch
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
import ast
import operator as op
import re

# Load models
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Functions (use your existing ones but remove print(), return strings instead)

@st.cache_resource
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

@st.cache_data
def load_chunks(chunks_file_path):
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        return f.read().split("\n---\n")

def route_query(query):
    query_lower = query.lower()
    math_keywords = ["calculate", "solve", "derivative", "integrate", "limit", "equation"]
    if any(word in query_lower for word in math_keywords):
        return "math"
    if query_lower.startswith("define") or "meaning of" in query_lower or "definition of" in query_lower:
        return "dictionary"
    return "rag"



# Supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}

def safe_eval(expr):
    def _eval(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            return operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):  # - <number>
            return operators[type(node.op)](_eval(node.operand))
        else:
            raise TypeError("Unsupported expression")

    try:
        parsed = ast.parse(expr, mode='eval')
        return _eval(parsed.body)
    except Exception as e:
        return f"Error: {str(e)}"

def handle_math(query):
    try:
        match = re.search(r'calculate\s+(.*)', query, re.IGNORECASE)
        if match:
            expression = match.group(1)
            result = safe_eval(expression)
            return f"Math Result: {result}"
        return "No expression found to calculate."
    except Exception:
        return "Sorry, couldn't calculate that."


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

def retrieve_top_chunks(query, k, index, chunks):
    query_vec = sentence_model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def refine_answer(query, chunks):
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

    prompt = f"Based on this context:\n{context}\n\nQuestion: {query}\nProvide a concise answer:"
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
        return final_answer, context
    except Exception:
        return "Answer generation failed.", context


index = load_faiss_index("embeddings/faiss_index.idx")
chunks = load_chunks("embeddings/chunks.txt")

# Streamlit UI
st.title(" RAG-Powered Multi-Agent Knowledge Assistant")
query = st.text_input("Ask your question:")

if query:
    route = route_query(query)
    st.markdown(f"###  Tool Selected: `{route.upper()}`")

    if route == "math":
        st.success(handle_math(query))
    elif route == "dictionary":
        st.info(handle_dictionary(query))
    else:
        top_chunks = retrieve_top_chunks(query, 3, index, chunks)
        answer, context = refine_answer(query, top_chunks)
        
        with st.expander(" Top Retrieved Context"):
            for i, chunk in enumerate(top_chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk.strip()}")
        
        st.success(f"**Answer:** {answer}")
