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
# UI STYLING
# =========================================================
# IMPORTANT:
# This UI does NOT use custom HTML div/span elements.
# Only CSS is used here.

st.markdown(
    """
    <style>

    /* ================================
       MAIN APP
       ================================ */

    .stApp {
        background-color: #f5f7fb;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ================================
       TEXT
       ================================ */

    h1, h2, h3, h4, h5, h6 {
        color: #172033 !important;
    }

    p {
        color: #344054;
    }

    /* ================================
       SIDEBAR
       ================================ */

    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e4e7ec;
    }

    section[data-testid="stSidebar"] p {
        color: #344054 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #172033 !important;
    }

    /* ================================
       FILE UPLOADER
       ================================ */

    [data-testid="stFileUploader"] {
        background-color: #f8faff !important;
        border: 2px dashed #b9c8e5 !important;
        border-radius: 14px !important;
        padding: 10px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #315fd3 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }

    /* ================================
       CHAT INPUT
       ================================ */

    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #d0d5dd !important;
        border-radius: 16px !important;
        box-shadow: 0 5px 20px rgba(16, 24, 40, 0.08) !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #172033 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #98a2b3 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #315fd3 !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* ================================
       CHAT MESSAGES
       ================================ */

    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        border: 1px solid #e4e7ec !important;
    }

    /* ================================
       BUTTONS
       ================================ */

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #cfd8ea !important;
        background-color: #eef3ff !important;
        color: #2856c5 !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background-color: #e0e9ff !important;
        border-color: #9fb5e5 !important;
    }

    /* ================================
       METRICS
       ================================ */

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(16, 24, 40, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #667085 !important;
    }

    [data-testid="stMetricValue"] {
        color: #315fd3 !important;
    }

    /* ================================
       ALERTS
       ================================ */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
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
# EMBEDDING MODEL
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
# PDF TEXT EXTRACTION
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

    st.title("📄 HR Document")

    st.caption(
        "Upload your organization's HR policy "
        "to begin."
    )

    st.subheader("📤 Upload Policy")

    uploaded_file = st.file_uploader(
        "Choose an HR policy PDF",
        type=["pdf"],
        help="Upload a text-based HR policy PDF.",
    )

    st.divider()

    st.subheader("🔎 RAG Pipeline")

    st.markdown(
        "🟦 **01 — PDF Upload**"
    )

    st.markdown(
        "🟦 **02 — PyMuPDF Extraction**"
    )

    st.markdown(
        "🟦 **03 — Text Chunking**"
    )

    st.markdown(
        "🟦 **04 — Sentence Transformers**"
    )

    st.markdown(
        "🟦 **05 — FAISS Retrieval**"
    )

    st.markdown(
        "🟦 **06 — Groq Generation**"
    )

    if st.session_state.document_name:

        st.divider()

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
# MAIN HEADER
# =========================================================

st.title(
    "📋 HR Policy Assistant"
)

st.caption(
    "AI-powered document question answering "
    "using Retrieval-Augmented Generation (RAG)."
)

st.markdown(
    "Upload an HR policy PDF and ask questions "
    "about its contents in natural language."
)

st.divider()


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.status(
            "Processing HR policy...",
            expanded=True
        ):

            st.write(
                "📖 Extracting text with PyMuPDF..."
            )

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

            st.write(
                f"✓ Extracted {len(pages)} pages"
            )

            st.write(
                "✂️ Creating text chunks..."
            )

            chunks = create_chunks(
                pages
            )

            st.write(
                f"✓ Created {len(chunks)} chunks"
            )

            st.write(
                "🧠 Creating Sentence Transformer embeddings..."
            )

            model = load_embedding_model()

            st.write(
                "🔎 Building FAISS vector index..."
            )

            index = create_faiss_index(
                chunks,
                model
            )

            st.write(
                "✓ FAISS index ready"
            )

        st.session_state.chunks = chunks

        st.session_state.index = index

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.messages = []

        st.success(
            "✅ HR policy uploaded and indexed successfully!"
        )


# =========================================================
# DOCUMENT STATISTICS
# =========================================================

if st.session_state.index is not None:

    st.subheader(
        "📊 Document Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="📄 Policy Chunks",
            value=len(
                st.session_state.chunks
            ),
        )

    with col2:

        st.metric(
            label="🔎 Vector Search",
            value="FAISS",
        )

    with col3:

        st.metric(
            label="🤖 Answer Generation",
            value="RAG",
        )

    st.success(
        "✓ Policy Ready — "
        "Ask a question about the uploaded HR policy below."
    )

else:

    st.info(
        "📤 Upload an HR policy PDF from the sidebar "
        "to start asking questions."
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

if st.session_state.index is not None:

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
                    "🔎 Searching the HR policy..."
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

                st.subheader(
                    "🤖 AI Answer"
                )

                st.markdown(
                    answer
                )

                # =========================================
                # SOURCES
                # =========================================

                st.divider()

                st.subheader(
                    "📚 Retrieved Sources"
                )

                seen_pages = set()

                for source in retrieved_chunks:

                    page = source["page"]

                    if page in seen_pages:
                        continue

                    seen_pages.add(page)

                    preview = (
                        source["text"][:350]
                        .replace("\n", " ")
                    )

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"**📄 Policy Page {page}**"
                        )

                        st.caption(
                            preview + "..."
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:

                st.error(
                    "Something went wrong:"
                )

                st.code(
                    str(error)
                )
