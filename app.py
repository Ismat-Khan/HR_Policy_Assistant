import os
import html

import faiss
import fitz
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

    /* =====================================================
       GLOBAL APP
    ===================================================== */

    .stApp {
        background-color: #f4f7fc;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* =====================================================
       HEADINGS
    ===================================================== */

    h1, h2, h3, h4, h5, h6 {
        color: #172033 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #172033 !important;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e3e8f2;
    }

    section[data-testid="stSidebar"] * {
        color: #172033 !important;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 800;
        color: #172033 !important;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        color: #667085 !important;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 18px;
    }


    /* =====================================================
       HERO
    ===================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #eef4ff 0%,
            #f8faff 55%,
            #edf2ff 100%
        );

        border: 1px solid #dce5f7;
        border-radius: 24px;

        padding: 30px 34px;
        margin-bottom: 25px;

        box-shadow: 0 8px 30px rgba(35, 75, 140, 0.06);
    }

    .hero-badge {
        display: inline-block;
        background-color: #e0eaff;
        color: #2856c5 !important;
        border-radius: 50px;
        padding: 6px 13px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #172033 !important;
        margin: 0;
    }

    .hero-subtitle {
        color: #667085 !important;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 8px;
    }


    /* =====================================================
       UPLOAD CARD
    ===================================================== */

    .upload-card {
        background-color: #ffffff;
        border: 1px solid #dfe6f2;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 10px;
        box-shadow: 0 5px 20px rgba(30, 60, 100, 0.04);
    }

    .upload-title {
        color: #172033 !important;
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .upload-description {
        color: #667085 !important;
        font-size: 13px;
    }


    /* =====================================================
       FILE UPLOADER
    ===================================================== */

    [data-testid="stFileUploader"] {
        background-color: #f7f9fd !important;
        border: 2px dashed #c9d6ed !important;
        border-radius: 14px !important;
        padding: 8px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }

    [data-testid="stFileUploader"] * {
        color: #344054 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #f7f9fd !important;
    }


    /* =====================================================
       RAG PIPELINE
    ===================================================== */

    .pipeline-card {
        background-color: #f8faff;
        border: 1px solid #e1e7f2;
        border-radius: 16px;
        padding: 18px;
    }

    .pipeline-heading {
        font-size: 17px;
        font-weight: 800;
        color: #172033 !important;
        margin-bottom: 12px;
    }

    .pipeline-step {
        background-color: #ffffff;
        border: 1px solid #e1e7f2;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    .pipeline-number {
        color: #315fd3 !important;
        font-weight: 800;
        margin-right: 8px;
    }

    .pipeline-text {
        color: #344054 !important;
        font-size: 13px;
    }


    /* =====================================================
       STAT CARDS
    ===================================================== */

    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e3e8f2;
        border-radius: 18px;
        padding: 20px;
        min-height: 115px;
        box-shadow: 0 5px 20px rgba(30, 60, 100, 0.04);
    }

    .stat-icon {
        font-size: 22px;
        margin-bottom: 7px;
    }

    .stat-value {
        font-size: 26px;
        font-weight: 800;
        color: #2856c5 !important;
    }

    .stat-label {
        color: #667085 !important;
        font-size: 13px;
        margin-top: 3px;
    }


    /* =====================================================
       POLICY READY
    ===================================================== */

    .ready-card {
        background: linear-gradient(
            135deg,
            #eef5ff,
            #f6f9ff
        );

        border: 1px solid #cddcf5;
        border-left: 5px solid #3b72df;

        border-radius: 16px;

        padding: 16px 20px;
        margin: 20px 0;
    }

    .ready-title {
        color: #214caa !important;
        font-weight: 750;
        font-size: 15px;
    }

    .ready-text {
        color: #667085 !important;
        font-size: 13px;
        margin-top: 3px;
    }


    /* =====================================================
       CHAT INPUT
    ===================================================== */

    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #d7e0ef !important;
        border-radius: 18px !important;
        padding: 5px !important;
        box-shadow: 0 6px 22px rgba(30, 60, 100, 0.08);
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #172033 !important;
        border: none !important;
        font-size: 15px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #98a2b3 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #315fd3 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] button svg {
        color: #ffffff !important;
    }


    /* =====================================================
       CHAT MESSAGES
    ===================================================== */

    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e3e8f2 !important;
        border-radius: 16px !important;
        margin-bottom: 12px !important;
        padding: 15px !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong {
        color: #172033 !important;
    }


    /* =====================================================
       ANSWER CARD
    ===================================================== */

    .answer-card {
        background-color: #ffffff;
        border: 1px solid #dfe6f2;
        border-radius: 20px;
        padding: 24px;
        margin-top: 10px;
        box-shadow: 0 8px 28px rgba(30, 60, 100, 0.06);
    }

    .answer-header {
        color: #2856c5 !important;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .answer-text {
        color: #172033 !important;
        font-size: 15px;
        line-height: 1.75;
    }

    .answer-text p,
    .answer-text li,
    .answer-text strong {
        color: #172033 !important;
    }


    /* =====================================================
       SOURCE CARDS
    ===================================================== */

    .sources-title {
        color: #172033 !important;
        font-size: 18px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .source-card {
        background-color: #ffffff;
        border: 1px solid #e1e7f0;
        border-radius: 14px;
        padding: 15px 18px;
        margin-bottom: 10px;
    }

    .source-page {
        color: #2856c5 !important;
        font-size: 13px;
        font-weight: 750;
    }

    .source-text {
        color: #667085 !important;
        font-size: 13px;
        line-height: 1.5;
        margin-top: 5px;
    }


    /* =====================================================
       BUTTONS
    ===================================================== */

    .stButton > button {
        background-color: #eef3ff !important;
        color: #2856c5 !important;
        border: 1px solid #ccd9f3 !important;
        border-radius: 10px !important;
        font-weight: 650 !important;
    }

    .stButton > button:hover {
        background-color: #e1eaff !important;
        border-color: #b7c9eb !important;
    }


    /* =====================================================
       ALERTS
    ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] p {
        color: #172033 !important;
    }

    </style>
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
# LOAD SENTENCE TRANSFORMER
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# =========================================================
# GROQ CLIENT
# =========================================================

def get_groq_client():

    api_key = None

    try:
        api_key = st.secrets.get(
            "GROQ_API_KEY"
        )
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv(
            "GROQ_API_KEY"
        )

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


# =========================================================
# EXTRACT TEXT FROM PDF
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
# CREATE TEXT CHUNKS
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

            chunk_words = words[
                start:end
            ]

            if chunk_words:

                chunks.append(
                    {
                        "text": " ".join(
                            chunk_words
                        ),
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

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

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
        question_embedding.astype(
            "float32"
        )
    )

    scores, indices = index.search(
        question_embedding,
        min(
            top_k,
            len(chunks)
        )
    )

    results = []

    for score, position in zip(
        scores[0],
        indices[0]
    ):

        if position == -1:
            continue

        results.append(
            {
                "text": chunks[position][
                    "text"
                ],
                "page": chunks[position][
                    "page"
                ],
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
3. If the answer is not found in the
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

Answer using only the provided HR policy context.
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

    return (
        response.choices[0]
        .message.content
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            📄 HR Document
        </div>

        <div class="sidebar-subtitle">
            Upload your organization's HR policy
            to begin.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-card">

            <div class="upload-title">
                📤 Upload Policy
            </div>

            <div class="upload-description">
                PDF files only
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="pipeline-card">

            <div class="pipeline-heading">
                🔎 RAG Pipeline
            </div>

            <div class="pipeline-step">
                <span class="pipeline-number">
                    01
                </span>
                <span class="pipeline-text">
                    PDF Upload
                </span>
            </div>

            <div class="pipeline-step">
                <span class="pipeline-number">
                    02
                </span>
                <span class="pipeline-text">
                    PyMuPDF Extraction
                </span>
            </div>

            <div class="pipeline-step">
                <span class="pipeline-number">
                    03
                </span>
                <span class="pipeline-text">
                    Text Chunking
                </span>
            </div>

            <div class="pipeline-step">
                <span class="pipeline-number">
                    04
                </span>
                <span class="pipeline-text">
                    Sentence Transformers
                </span>
            </div>

            <div class="pipeline-step">
                <span class="pipeline-number">
                    05
                </span>
                <span class="pipeline-text">
                    FAISS Retrieval
                </span>
            </div>

            <div class="pipeline-step">
                <span class="pipeline-number">
                    06
                </span>
                <span class="pipeline-text">
                    Groq Generation
                </span>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.document_name:

        st.markdown("---")

        st.success(
            f"📄 {st.session_state.document_name}"
        )

        st.caption(
            f"{len(st.session_state.chunks)} "
            "policy chunks indexed"
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
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            AI-POWERED HR KNOWLEDGE
        </div>

        <div class="hero-title">
            📋 HR Policy Assistant
        </div>

        <div class="hero-subtitle">
            Upload an HR policy document and ask questions
            in natural language. RAG retrieves relevant
            policy sections before generating the answer.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PROCESS UPLOADED PDF
# =========================================================

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "📖 Reading HR policy..."
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
            "🧠 Building FAISS search index..."
        ):

            model = (
                load_embedding_model()
            )

            index = create_faiss_index(
                chunks,
                model
            )

        st.session_state.chunks = chunks

        st.session_state.index = index

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.messages = []

        st.success(
            "✅ Policy uploaded and ready!"
        )


# =========================================================
# DOCUMENT STATISTICS
# =========================================================

if st.session_state.index is not None:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="stat-card">

                <div class="stat-icon">
                    📄
                </div>

                <div class="stat-value">
                    {len(st.session_state.chunks)}
                </div>

                <div class="stat-label">
                    Policy Chunks
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="stat-card">

                <div class="stat-icon">
                    🔎
                </div>

                <div class="stat-value">
                    FAISS
                </div>

                <div class="stat-label">
                    Vector Search
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="stat-card">

                <div class="stat-icon">
                    🤖
                </div>

                <div class="stat-value">
                    RAG
                </div>

                <div class="stat-label">
                    AI Answer Generation
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="ready-card">

            <div class="ready-title">
                ✓ Policy Ready
            </div>

            <div class="ready-text">
                Your document has been indexed.
                Ask a question below to retrieve
                relevant HR policy information.
            </div>

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
        "📤 Upload an HR policy PDF from the "
        "sidebar to start asking questions."
    )

else:

    question = st.chat_input(
        "Ask something about the HR policy..."
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

                model = (
                    load_embedding_model()
                )

                with st.spinner(
                    "🔎 Searching policy..."
                ):

                    retrieved_chunks = (
                        retrieve_chunks(
                            question,
                            st.session_state.index,
                            st.session_state.chunks,
                            model,
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

                # =========================================
                # ANSWER
                # =========================================

                st.markdown(
                    """
                    <div class="answer-card">

                        <div class="answer-header">
                            🤖 AI Answer
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(answer)

                # =========================================
                # SOURCES
                # =========================================

                st.markdown(
                    """
                    <div class="sources-title">
                        📚 Retrieved Sources
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                seen_pages = set()

                for source in retrieved_chunks:

                    page = source["page"]

                    if page in seen_pages:
                        continue

                    seen_pages.add(page)

                    preview = html.escape(
                        source["text"][:350]
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-page">
                                📄 Policy Page {page}
                            </div>

                            <div class="source-text">
                                {preview}...
                            </div>

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
