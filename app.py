import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import re
import pkg_resources
from symspellpy import SymSpell, Verbosity

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Load a dictionary
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_dictionary("/home/cdot/.config/JetBrains/PyCharmCE2022.1/scratches/NLP/custom_dictionary.txt", term_index=0, count_index=1)

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

documents = [
    "Enter your context documentes here"
]

@st.cache_resource
def load_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(documents)
    hnsw_index = faiss.IndexHNSWFlat(doc_embeddings.shape[1], 32)
    hnsw_index.add(np.array(doc_embeddings).astype("float32"))
    return model, hnsw_index

model, hnsw_index = load_data()

def preprocess_text(text):
    # Use regex to split text into tokens, preserving numeric/alphanumeric data
    tokens = re.findall(r'\w+|\d+\w*|\S+', text)
    return tokens

# Function to correct spelling while preserving numeric data
def correct_spelling(text):
    # Split text into tokens
    tokens = preprocess_text(text)
    corrected_tokens = []
    # print(tokens)
    for token in tokens:
        # If the token is numeric or alphanumeric, preserve it
        if token.isdigit() or re.match(r'\d+\w*', token) or re.match(r'[.,]', token):# or re.match(r'\S+', token):
            corrected_tokens.append(token)
        else:
            # Otherwise, correct the token using SymSpell
            suggestions = sym_spell.lookup(token, max_edit_distance=2, verbosity=1)
            # print(suggestions)
            if suggestions:
                corrected_token = suggestions[0].term  # Use the best suggestion
            else:
                corrected_token = token  # If no suggestion, keep the original token
            corrected_tokens.append(corrected_token)

    # Join the corrected tokens into a sentence
    return " ".join(corrected_tokens)

def rag_qa(question):
    question = correct_spelling(question)
    print(f'correct_question : {question}')
    question_embedding = model.encode([question])
    distances, retrieved_indices = hnsw_index.search(np.array(question_embedding).astype("float32"), k=1)
    retrieved_doc = documents[retrieved_indices[0][0]]
    prompt = f"Context: {retrieved_doc}\n\nQ: {question}\nA: If the answer is unclear from the context, respond with 'I don't know'"
    response = qa_pipeline(prompt, max_length=500)
    answer = response[0]['generated_text']
    print(f'generated_answer : {answer}')
    return answer

# Step 4: Streamlit UI Implementation
st.title("🧠 Ask anything about Ginni !")
question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if question.strip():
        answer = rag_qa(question)
        st.success(f"**Answer:** {answer}")
    else:
        st.warning("Please enter a valid question.")