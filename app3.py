import streamlit as st
import os
import time
import hashlib
import mysql.connector
import json
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import pyttsx3
import base64
import io
from pdf2image import convert_from_path
import fitz
import re
from keybert import KeyBERT

# --- Load environment variables ---
load_dotenv()
groq_api_key = os.getenv('API_KEY')

# --- MySQL Connection ---
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Iampk@15",
    database="qa"
)
cursor = db.cursor()

# --- Streamlit Page Config ---
st.set_page_config(page_title="📄 Document QA System", layout="wide")

# --- Page styling ---
st.markdown("""
    <style>
    .main { background-color: #1f2937; color: white; }
    h1, h2, h3 { color: #f9fafb; font-weight: bold; }
    .stButton>button {
        color: white;
        background-color: #6366f1;
        border: none;
        padding: 12px 24px;
        margin: 10px 5px;
        font-size: 16px;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    }
    .stButton>button:hover {
        background-color: #4f46e5;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        background-color: #374151;
        color: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #6b7280;
    }
    .stExpander {
        background-color: #374151;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
        margin-bottom: 20px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Load model once
keyword_model = KeyBERT(model='all-MiniLM-L6-v2')

def extract_dynamic_keyword(question):
    """Extract top keyword or phrase from the user's question."""
    keywords = keyword_model.extract_keywords(question, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
    if keywords:
        return keywords[0][0]  # Return the top keyword
    return None

def preview_pdf(file_path):
    # Convert the first page of the PDF to an image
    images = convert_from_path(file_path, first_page=1, last_page=1)  # Limit to 1 page for preview
    img = images[0]
    
    # Save image temporarily to display in Streamlit
    img_path = "temp_preview.png"
    img.save(img_path, 'PNG')
    
    # Display the image in Streamlit
    st.image(img_path, caption="Preview of Uploaded PDF", use_column_width=True)
    os.remove(img_path)  # Clean up temporary image file
    

def extract_relevant_snippet(text, keyword="Article 2", window=500):
    """Extract a small snippet around the keyword."""
    match = re.search(re.escape(keyword), text, re.IGNORECASE)
    if match:
        start = max(0, match.start() - window//2)
        end = min(len(text), match.end() + window//2)
        return text[start:end]
    return text  # fallback: return full text
   
def highlight_text_in_pdf_and_find_page(pdf_path, text_to_highlight):
    doc = fitz.open(pdf_path)
    found_page_num = None
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        found_instances = page.search_for(text_to_highlight)
        
        if found_instances:
            for inst in found_instances:
                page.add_highlight_annot(inst)
            found_page_num = page_num  # Save the page number where highlight happened
            break  # Only highlight first found occurrence
    
    # Save highlighted file
    highlighted_pdf_path = "highlighted_temp.pdf"
    doc.save(highlighted_pdf_path)
    doc.close()
    
    return highlighted_pdf_path, found_page_num

# --- Helper functions ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def generate_audio_clip(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)  # Speed (default 200)
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

    # List voices
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Choose a voice (0: male, 1: female usually)

    audio_buffer = io.BytesIO()
    engine.save_to_file(text, 'temp_audio.mp3')
    engine.runAndWait()

    # Read the generated file back into memory
    with open('temp_audio.mp3', 'rb') as f:
        audio_buffer.write(f.read())

    audio_buffer.seek(0)
    return audio_buffer

def show_lottie_popup(animation_url, height=300, duration=3):
    animation_json = load_lottieurl(animation_url)
    if animation_json:
        container = st.empty()
        with container:
            st_lottie(animation_json, height=height, speed=1)
            time.sleep(duration)
        container.empty()

def vector_embedding(file_path, doc_hash, filename):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    # add metadata
    for doc in final_documents:
        if not hasattr(doc, "metadata"):
            doc.metadata = {}
        doc.metadata['document_hash'] = doc_hash
        doc.metadata['filename'] = filename

    vectors = FAISS.from_documents(final_documents, embedding_model)

    # IMPORTANT: Save this vectorstore mapped to its document hash
    st.session_state.setdefault('vectorstores', {})  # Make sure dict exists
    st.session_state['vectorstores'][doc_hash] = vectors



def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        return hashlib.sha256(file_bytes).hexdigest()

def search_database_semantic(doc_hash, user_question):
    user_embedding = embedding_model.embed_query(user_question)
    cursor.execute("SELECT question, question_embedding, answer FROM qa_table WHERE document_hash = %s", (doc_hash,))
    records = cursor.fetchall()

    best_similarity = 0
    best_answer = None

    for q, q_embed_json, a in records:
        q_embedding = np.array(json.loads(q_embed_json))
        sim = cosine_similarity([user_embedding], [q_embedding])[0][0]

        if sim > best_similarity:
            best_similarity = sim
            best_answer = a

    if best_similarity >= 0.8:
        return best_answer
    else:
        return None

def save_to_database(doc_hash, filename, question, answer):
    try:
        question_embedding = embedding_model.embed_query(question)
        question_embedding_json = json.dumps(question_embedding)
        cursor.execute(
            "INSERT INTO qa_table (document_hash, filename, question, question_embedding, answer) VALUES (%s, %s, %s, %s, %s)",
            (doc_hash, filename, question, question_embedding_json, answer)
        )
        db.commit()
    except Exception as e:
        print(f"❌ Failed to save to DB: {e}")


# --- LLM and Embeddings ---
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

prompt = ChatPromptTemplate.from_template(
    """
    You are an intelligent assistant. Based on the following context, answer the user's question.

    Context:
    {context}

    Question:
    {input}

    If the answer is not in the context or not a valid question, just say "I don't know."
    """
)

# --- Lottie Animations URLs ---
upload_animation_url = "https://assets10.lottiefiles.com/packages/lf20_4kx2q32n.json"
ingest_animation_url = "https://assets2.lottiefiles.com/packages/lf20_u4yrau.json"
thinking_animation_url = "https://assets9.lottiefiles.com/packages/lf20_usmfx6bp.json"

# --- App Layout ---
st.title("📄 Document Question Answering System")
st.caption("Upload your PDF ➔ Ingest ➔ Ask your questions!")

uploaded_files = st.file_uploader("📤 Upload your PDF files", type=["pdf"], accept_multiple_files=True)


if uploaded_files:
    for uploaded_file in uploaded_files:
        st.success(f"✅ Uploaded: {uploaded_file.name}")

        # Save uploaded file temporarily
        saved_file_path = f"temp_{uploaded_file.name}"
        with open(saved_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Calculate document hash
        doc_hash = get_file_hash(saved_file_path)
        st.session_state.setdefault('documents', {})
        st.session_state.documents[doc_hash] = uploaded_file.name

        # Check if document already exists in database
        cursor.execute("SELECT filename FROM qa_table WHERE document_hash = %s LIMIT 1", (doc_hash,))
        result = cursor.fetchone()

        # Always set document_hash in session (important!)
        st.session_state.document_hash = doc_hash

        if result:
            old_filename = result[0]
            if old_filename != uploaded_file.name:
                cursor.execute("UPDATE qa_table SET filename = %s WHERE document_hash = %s", (uploaded_file.name, doc_hash))
                db.commit()
                st.info(f"🔄 Updated filename in DB from {old_filename} to {uploaded_file.name}")
            else:
                st.info(f"📂 Document already ingested previously as {uploaded_file.name}")

            # Rebuild vectors even for already-ingested document
            vector_embedding(saved_file_path, doc_hash, uploaded_file.name)

        else:
            # New document: Ingest and store vectors
            vector_embedding(saved_file_path, doc_hash, uploaded_file.name)
            st.success(f"✅ Ingested and stored: {uploaded_file.name}")

        # --- 🆕 Only AFTER successful ingestion, show Preview Button ---
        with st.expander(f"👀 Preview {uploaded_file.name} (optional)", expanded=False):
            if st.button(f"Show Preview of {uploaded_file.name}", key=f"preview_button_{uploaded_file.name}"):
                preview_pdf(saved_file_path)


if st.session_state.get('vectorstores', {}):
    with st.expander("📝 Ask a Question", expanded=True):
        uploaded_documents = st.session_state.get('documents', {})

        if uploaded_documents:
            selected_filename = st.selectbox(
                "📄 Select a document to ask questions about:",
                list(uploaded_documents.values()),
                key="ask_doc_select"
            )

            selected_hash = None
            for hash_val, file_name in uploaded_documents.items():
                if file_name == selected_filename:
                    selected_hash = hash_val
                    break

            question = st.text_input("Type your question about the selected document...", key="ask_question_input")

            if st.button("📨 Submit", key="ask_submit_button"):
                if question.strip() == "":
                    st.error("Please enter a valid question!")
                else:
                    try:
                        show_lottie_popup(thinking_animation_url, height=250, duration=2)

                        # 1️⃣ First, check if similar question exists in database
                        cached_answer = search_database_semantic(selected_hash, question)

                        if cached_answer:
                            st.success("✅ Found a similar question in database!")
                            answer = cached_answer
                            from_cache = True
                        else:
                            # 2️⃣ Else, retrieve from document and call LLM
                            selected_vectorstore = st.session_state.vectorstores[selected_hash]
                            retriever = selected_vectorstore.as_retriever()
                            retrieved_docs = retriever.get_relevant_documents(question)
                            st.session_state['retrieved_docs'] = retrieved_docs

                            if not retrieved_docs:
                                st.warning("No relevant documents found!")
                                answer = None
                                from_cache = False
                            else:
                                start = time.process_time()  # Start timing here
                                document_chain = create_stuff_documents_chain(llm, prompt)
                                response = document_chain.invoke({'input': question, 'context': retrieved_docs})
                                answer = response
                                from_cache = False
                                
                                normalized_answer = answer.strip().lower()

                                if ("i don't know" in normalized_answer 
                                    or "couldn't find" in normalized_answer 
                                    or normalized_answer == "" 
                                    or "no relevant information" in normalized_answer):
                                    st.warning("⚠️ No meaningful answer found in document. Not saving this question.")
                                    answer = None
                                else:
                                    save_to_database(selected_hash, selected_filename, question, answer)

                        # 3️⃣ Save the answer if available
                        if answer:
                            st.session_state['last_answer'] = answer
                            st.session_state['last_selected_filename'] = selected_filename
                            st.session_state['last_from_cache'] = from_cache
                            if not from_cache:
                                st.session_state['last_response_time'] = time.process_time() - start
                            else:
                                st.session_state['last_response_time'] = None

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        # 🚀 After button (outside), always check if last_answer exists to re-display
        if 'last_answer' in st.session_state and st.session_state['last_answer']:
            last_answer = st.session_state['last_answer']
            last_filename = st.session_state.get('last_selected_filename', "Unknown Document")
            from_cache = st.session_state.get('last_from_cache', False)
            last_response_time = st.session_state.get('last_response_time', None)

            if not from_cache and last_response_time is not None:
                st.success(f"✅ Response time: {last_response_time:.2f} seconds")
                st.info("🤖 Generated fresh answer for you")
            else:
                st.info("🧠 Retrieved instant answer from saved questions")

            st.write("📝 **Answer:**")
            st.caption(f"📄 Answer from: {last_filename}")
            st.write(last_answer)

            # 🎤 Voice playback for previous answer
            if st.checkbox("🔊 Listen to the Answer", key="listen_to_answer_checkbox_last"):
                audio_clip = generate_audio_clip(last_answer)
                st.audio(audio_clip, format="audio/mp3")

            try:
                # Dynamic Keyword extraction from question
                dynamic_keyword = extract_dynamic_keyword(question)
                retrieved_docs = st.session_state.get('retrieved_docs', None)
                
                if retrieved_docs:
                    best_doc = retrieved_docs[0]
                    full_text = best_doc.page_content

                    # Extract focused snippet based on dynamic keyword
                    if dynamic_keyword:
                        context_text = extract_relevant_snippet(full_text, keyword=dynamic_keyword)
                    else:
                        context_text = full_text
                else:
                    context_text = ""

                highlighted_pdf, highlighted_page_num = highlight_text_in_pdf_and_find_page(f"temp_{selected_filename}", context_text)

                if highlighted_page_num is not None:
                    with st.spinner('Generating highlighted preview...'):
                        images = convert_from_path(highlighted_pdf, first_page=highlighted_page_num + 1, last_page=highlighted_page_num + 1)
                        img = images[0]
                        st.image(img, caption=f"📄 Highlighted Preview: Page {highlighted_page_num + 1}", use_column_width=True)
                else:
                    st.warning("❗ Could not find matching text to highlight in document.")
            except Exception as e:
                st.warning(f"⚠️ Could not highlight or preview: {str(e)}")





# --- Recently Asked Questions Section ---

with st.expander("🕑 Recently Asked Questions on Your Documents", expanded=True):
    uploaded_documents = st.session_state.get('documents', {})

    if uploaded_documents:
        selected_filename = st.selectbox(
            "📄 Select a document to view recent questions:",
            list(uploaded_documents.values()),
            key="recent_doc_select"
        )

        selected_hash = None
        for hash_val, file_name in uploaded_documents.items():
            if file_name == selected_filename:
                selected_hash = hash_val
                break

        if selected_hash:
            cursor.execute(
                "SELECT question, answer FROM qa_table WHERE document_hash = %s ORDER BY id DESC LIMIT 10",
                (selected_hash,)
            )
            recent_qas = cursor.fetchall()

            if recent_qas:
                for idx, (q, a) in enumerate(recent_qas):
                    with st.container():
                        st.markdown(f"**🔹 {q}**", unsafe_allow_html=True)

                        if st.button(f"Show Answer {idx+1}", key=f"recent_show_{idx}"):
                            # Save into session state
                            st.session_state['recent_answer'] = a
                            st.session_state['recent_question'] = q
                            st.session_state['recent_filename'] = selected_filename
                            st.session_state['recent_idx'] = idx

                # 🔥 After buttons loop, check if a recent_answer exists
                if 'recent_answer' in st.session_state and st.session_state['recent_answer']:
                    st.success("✅ Answer:")
                    st.caption(f"📄 Answer from: {st.session_state.get('recent_filename', 'Unknown Document')}")
                    st.write(st.session_state['recent_answer'])

                    # 🎤 Voice option
                    if st.checkbox(f"🔊 Listen to the Answer", key=f"listen_recent_answer"):
                        audio_clip = generate_audio_clip(st.session_state['recent_answer'])
                        st.audio(audio_clip, format="audio/mp3")

            else:
                st.info("No recent questions yet for this document. Ask your first question!")
        else:
            st.warning("Selected document not found.")
    else:
        st.info("Upload and ingest a document first to see recent questions.")

