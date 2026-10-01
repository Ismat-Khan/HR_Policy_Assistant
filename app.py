import os
import html

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

        /* =========================
           MAIN APP
        ========================= */

        .stApp {
            background-color: #f7f9fc;
            color: #172033;
        }

        /* =========================
           HEADINGS
        ========================= */

        h1, h2, h3, h4, h5, h6 {
            color: #172033 !important;
        }

        /* =========================
           MAIN TITLE
        ========================= */

        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #172033 !important;
            margin-bottom: 5px;
        }

        .subtitle {
            color: #667085 !important;
            font-size: 17px;
            margin-bottom: 25px;
        }

        /* =========================
           GENERAL MARKDOWN
        ========================= */

        [data-testid="stMarkdownContainer"] p {
            color: #172033 !important;
        }

        [data-testid="stMarkdownContainer"] li {
            color: #172033 !important;
        }

        [data-testid="stMarkdownContainer"] strong {
            color: #172033 !important;
        }

        [data-testid="stMarkdownContainer"] em {
            color: #172033 !important;
        }

        /* =========================
           CHAT MESSAGES
        ========================= */

        [data-testid="stChatMessage"] {
            color: #172033 !important;
        }

        [data-testid="stChatMessage"] p {
            color: #172033 !important;
        }

        [data-testid="stChatMessage"] li {
            color: #172033 !important;
        }

        [data-testid="stChatMessage"] strong {
            color: #172033 !important;
        }

        [data-testid="stChatMessage"] code {
            color: #172033 !important;
        }

        /* =========================
           STATUS CARD
        ========================= */

        .status-card {
            background-color: #eef7ff;
            color: #172033 !important;
            border-left: 5px solid #2563eb;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }

        .status-card b {
            color: #172033 !important;
        }

        /* =========================
           SOURCE CARD
        ========================= */

        .source-card {
            background-color: #ffffff;
            color: #172033 !important;
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            padding: 15px;
            margin-top: 10px;
        }

        .source-card b {
            color: #172033 !important;
        }

        .source-card span {
            color: #667085 !important;
        }

        /* =========================
           METRIC CARDS
        ========================= */

        .metric-card {
            background-color: #ffffff;
            color: #172033 !important;
            border: 1px solid #e4e7ec;
            border-radius: 14px;
            padding: 18px;
            text-align: center;
        }

        .metric-number {
            font-size: 28px;
            font-weight: 800;
            color: #2563eb !important;
        }

        .metric-label {
            color: #667085 !important;
            font-size: 14px;
        }

        /* =========================
           FILE UPLOADER
        ========================= */

        div[data-testid="stFileUploader"] {
            background-color: #ffffff;
            color: #172033 !important;
            border-radius: 14px;
            padding: 10px;
        }

        div[data-testid="stFileUploader"] * {
            color: #172033 !important;
        }

        /* =========================
           SIDEBAR
        ========================= */

        section[data-testid="stSidebar"] {
            background-color: #ffffff;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #172033 !important;
        }

        section[data-testid="stSidebar"] p {
            color: #172033 !important;
        }

        section[data-testid="stSidebar"] span {
            color: #172033 !important;
        }

        section[data-testid="stSidebar"] label {
            color: #172033 !important;
        }

        section[data-testid="stSidebar"] button {
            color: #172033 !important;
        }

        /* =========================
           INPUTS
        ========================= */

        input {
            color: #172033 !important;
            background-color: #ffffff !important;
        }

        textarea {
            color: #172033 !important;
            background-color: #ffffff !important;
        }

        /* =========================
           BUTTONS
        ========================= */

        button {
            color: #172033 !important;
        }

        /* =========================
           INFO / SUCCESS / ERROR
        ========================= */

        [data-testid="stAlert"] {
            color: #172033 !important;
        }

        [data-testid="stAlert"] p {
            color: #172033 !important;
        }

        /* =========================
           CODE BLOCKS
        ========================= */

        pre {
            color: #172033 !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TITLE
# =========================================================

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


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# GROQ CLIENT
# =========================================================

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


# =========================================================
# EXTRACT PDF TEXT
# =========================================================

def extract_pdf_text(pdf_bytes):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

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


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(
    pages,
    chunk_size=900,
    overlap=150
):

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

                chunk_text = " ".join(
                    chunk_words
                )

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


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(
    chunks,
    model
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    model,
    top_k=4
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_embedding = (
        question_embedding.astype("float32")
    )

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

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


# =========================================================
# ASK GROQ
# =========================================================

def ask_groq(
    question,
    retrieved_chunks
):

    client = get_groq_client()

    if client is None:

        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"[Page {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are an HR Policy Assistant.

Answer the user's question using ONLY
the HR policy context provided.

Rules:

1. Do not invent HR policies.
2. Do not use outside information.
3. If the answer is not present in the
   provided context, clearly say that the
   uploaded policy does not contain enough
   information to answer the question.
4. Keep the answer clear and professional.
5. Mention the relevant policy page when
   possible.
6. Do not make assumptions.
7. Do not provide legal advice.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question using only the
provided HR policy context.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

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


# =========================================================
# SIDEBAR
# =========================================================

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
            f"Loaded:\n\n"
            f"{st.session_state.document_name}"
        )

        st.caption(
            f"Chunks: "
            f"{len(st.session_state.chunks)}"
        )

        if st.button(
            "🗑️ Clear Document",
            use_container_width=True
        ):

            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.document_name = None
            st.session_state.messages = []

            st.rerun()


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "📖 Reading HR policy PDF..."
        ):

            pdf_bytes = (
                uploaded_file.getvalue()
            )

            pages = extract_pdf_text(
                pdf_bytes
            )

        if not pages:

            st.error(
                "No readable text was found "
                "in this PDF. Please upload a "
                "text-based PDF."
            )

            st.stop()

        with st.spinner(
            "✂️ Creating policy chunks..."
        ):

            chunks = create_chunks(
                pages
            )

        with st.spinner(
            "🧠 Creating embeddings and "
            "FAISS index..."
        ):

            embedding_model = (
                load_embedding_model()
            )

            index = create_faiss_index(
                chunks,
                embedding_model
            )

        st.session_state.chunks = chunks

        st.session_state.index = index

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.messages = []

        st.success(
            f"✅ {uploaded_file.name} "
            "is ready for questions!"
        )


# =========================================================
# DOCUMENT STATUS
# =========================================================

if st.session_state.index is not None:

    chunks_count = len(
        st.session_state.chunks
    )

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
            """
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
            """
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
            Ask questions about the uploaded
            HR policy below.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# QUESTION INPUT
# =========================================================

if st.session_state.index is None:

    st.info(
        "👈 Upload an HR policy PDF "
        "from the sidebar to begin."
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

                embedding_model = (
                    load_embedding_model()
                )

                with st.spinner(
                    "🔎 Searching the policy..."
                ):

                    retrieved_chunks = (
                        retrieve_chunks(
                            question,
                            st.session_state.index,
                            st.session_state.chunks,
                            embedding_model,
                            top_k=4,
                        )
                    )

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = ask_groq(
                        question,
                        retrieved_chunks
                    )

                # Answer heading
                st.markdown(
                    "### 🤖 Answer"
                )

                # Normal Streamlit markdown.
                # No custom HTML wrapper.
                st.markdown(answer)

                # Sources
                st.markdown(
                    "### 📚 Sources"
                )

                seen_pages = set()

                for source in retrieved_chunks:

                    page = source["page"]

                    if page in seen_pages:
                        continue

                    seen_pages.add(page)

                    preview = source["text"][:300]

                    preview = html.escape(
                        preview
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>📄 Page {page}</b>
                            <br><br>
                            <span>
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
                    f"Something went wrong:\n\n"
                    f"{error}"
                )
