import io
import os
import html

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f9fc;
        }

        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #172033;
            margin-bottom: 5px;
        }

        .subtitle {
            color: #667085;
            font-size: 17px;
            margin-bottom: 25px;
        }

        .info-card {
            background: white;
            padding: 22px;
            border-radius: 16px;
            border: 1px solid #e4e7ec;
            margin-bottom: 20px;
        }

        .status-card {
            background: #eef7ff;
            border-left: 5px solid #2563eb;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }

        .source-card {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            padding: 15px;
            margin-top: 10px;
        }

        .metric-card {
            background: white;
            border: 1px solid #e4e7ec;
            border-radius: 14px;
            padding: 18px;
            text-align: center;
        }

        .metric-number {
            font-size: 28px;
            font-weight: 800;
            color: #2563eb;
        }

        .metric-label {
            color: #667085;
            font-size: 14px;
        }

        div[data-testid="stFileUploader"] {
            background: white;
            border-radius: 14px;
            padding: 10px;
        }

        .answer-box {
            background: white;
            border: 1px solid #e4e7ec;
            border-radius: 16px;
            padding: 22px;
            margin-top: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📋 HR Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload an HR policy PDF and ask questions using
        Retrieval-Augmented Generation (RAG).
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# GET GROQ CLIENT
# ---------------------------------------------------------

def get_groq_client():
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ---------------------------------------------------------
# EXTRACT TEXT FROM PDF
# ---------------------------------------------------------

def extract_pdf_text(pdf_bytes):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    document.close()

    return pages


# ---------------------------------------------------------
# SPLIT TEXT INTO CHUNKS
# ---------------------------------------------------------

def create_chunks(pages, chunk_size=900, overlap=150):
    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        words = text.split()

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]

            if chunk_words:
                chunk_text = " ".join(chunk_words)

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# ---------------------------------------------------------
# CREATE FAISS INDEX
# ---------------------------------------------------------

def create_faiss_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# ---------------------------------------------------------

def retrieve_chunks(question, index, chunks, model, top_k=4):
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_embedding = question_embedding.astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue

        results.append(
            {
                "text": chunks[index_position]["text"],
                "page": chunks[index_position]["page"],
                "score": float(score),
            }
        )

    return results


# ---------------------------------------------------------
# ASK GROQ
# ---------------------------------------------------------

def ask_groq(question, retrieved_chunks):
    client = get_groq_client()

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Answer the user's question using ONLY the HR policy context
provided below.

Rules:
1. Do not invent policies.
2. If the answer is not present in the provided context,
   clearly say that the uploaded policy does not contain
   enough information to answer the question.
3. Keep the answer clear and professional.
4. Mention the relevant policy page when possible.
5. Do not treat your answer as legal advice.
6. Do not make assumptions about information that is not
   present in the document.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer based only on the policy context.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## 📄 Document")

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"],
        help="Upload a text-based HR policy PDF.",
    )

    st.markdown("---")

    st.markdown("### 🔎 RAG Pipeline")

    st.markdown(
        """
        **1. Upload**  
        HR policy PDF

        **2. Extract**  
        PyMuPDF

        **3. Chunk**  
        Split policy text

        **4. Embed**  
        Sentence Transformers

        **5. Search**  
        FAISS similarity search

        **6. Generate**  
        Groq LLM
        """
    )

    st.markdown("---")

    if st.session_state.document_name:

        st.success(
            f"Loaded:\n\n{st.session_state.document_name}"
        )

        st.caption(
            f"Chunks: {len(st.session_state.chunks)}"
        )

        if st.button("🗑️ Clear Document", use_container_width=True):

            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.document_name = None
            st.session_state.messages = []

            st.rerun()


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner("📖 Reading HR policy PDF..."):

            pdf_bytes = uploaded_file.getvalue()

            pages = extract_pdf_text(pdf_bytes)

        if not pages:
            st.error(
                "No readable text was found in this PDF. "
                "Please upload a text-based PDF."
            )
            st.stop()

        with st.spinner("✂️ Creating policy chunks..."):

            chunks = create_chunks(pages)

        with st.spinner(
            "🧠 Creating embeddings and FAISS index..."
        ):

            embedding_model = load_embedding_model()

            index = create_faiss_index(
                chunks,
                embedding_model,
            )

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.document_name = uploaded_file.name
        st.session_state.messages = []

        st.success(
            f"✅ {uploaded_file.name} is ready for questions!"
        )


# ---------------------------------------------------------
# DOCUMENT STATUS
# ---------------------------------------------------------

if st.session_state.index is not None:

    chunks_count = len(st.session_state.chunks)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {chunks_count}
                </div>
                <div class="metric-label">
                    Policy Chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    FAISS
                </div>
                <div class="metric-label">
                    Vector Search
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    RAG
                </div>
                <div class="metric-label">
                    Answer Generation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    st.markdown(
        """
        <div class="status-card">
            <b>✅ Policy Ready</b><br>
            Ask questions about the uploaded HR policy below.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------------

if st.session_state.index is None:

    st.info(
        "👈 Upload an HR policy PDF from the sidebar to begin."
    )

else:

    question = st.chat_input(
        "Ask a question about the HR policy..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                embedding_model = load_embedding_model()

                with st.spinner(
                    "🔎 Searching the policy..."
                ):

                    retrieved_chunks = retrieve_chunks(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        embedding_model,
                        top_k=4,
                    )

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = ask_groq(
                        question,
                        retrieved_chunks,
                    )

                st.markdown(
                    '<div class="answer-box">',
                    unsafe_allow_html=True,
                )

                st.markdown(answer)

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

                # Sources
                st.markdown("### 📚 Sources")

                seen_pages = set()

                for source in retrieved_chunks:

                    page = source["page"]

                    if page in seen_pages:
                        continue

                    seen_pages.add(page)

                    preview = source["text"][:300]

                    preview = html.escape(preview)

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>📄 Page {page}</b><br>
                            <span style="color:#667085;">
                                {preview}...
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:

                st.error(
                    f"Something went wrong:\n\n{error}"
                )
