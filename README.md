Assignment: Build a RAG-Powered Multi-Agent Q&A Assistant

A simple RAG-Powered Q&A Assistant that uses pipeline to answer the questions

Features
1) Can retrieve top 3 relevant chunks from document nlp_intro using FAISS
2) Can perform basic mathematical computations like +,-,*,/,**,%,unary signs using ast
3) Can retrieve definitions of terms using NLTK's wordnet
4) Agentic WorkFlow:Routes the query to the appropriate tool
5) All these features are implemented in a simple streamlit interface


Architecture:
Routing Agent
├──> Math Tool (ast)
├──> Dictionary Tool (WordNet)
└──> RAG Pipeline
├── Retrieve top 3 chunks using FAISS
└── Use GPT-2 to generate answer from context

Key Design Choices
FAISS Index: Fast vector-based retrieval from pre-embedded document chunks.
Sentence-Level Similarity: Improves relevance by refining retrieved chunks into granular context.
GPT-2: Lightweight, open-source text generator for concise answer generation.
Streamlit: Easy-to-use frontend for context visualization.

How to run the code:
1) This is a simple implementation of the RAG system which makes use of one document alone for now
2) nlp_intro is the document that has been used for the RAG system which is present under the data folder
3) Chunks from this pdf has been retrieved using the data_processor.py and converted to vectors using faiss_index.py
4) The main application is run through main.py, which includes
    -A dictionary tool that retrieves the definition if the query is of the form: define <any word>.
    -A math tool that returns the calculated result for queries like: calculate <num1> <operator> <num2>.
    -A RAG system that retrieves the top 3 relevant chunks and generates a refined answer using GPT-2, based on queries related to the nlp_intro document.
5) To launch the streamlit interface- run the following command,
streamlit run main.py

6) retriever_agent.py runs the same program but in CLI with a few restrictions (uses eval for math), so can see it for reference alone

