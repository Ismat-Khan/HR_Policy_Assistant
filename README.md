# 📋 HR Policy Assistant

An AI-powered HR Policy Assistant built with Retrieval-Augmented Generation (RAG).

Users can upload an HR policy PDF and ask questions about the document. The application retrieves the most relevant sections of the policy and sends them to a Groq-powered LLM to generate a grounded answer.

## 🚀 Features

- Upload HR policy PDF
- Extract text using PyMuPDF
- Split documents into smaller chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Retrieve relevant policy sections
- Generate answers using Groq API
- Display source pages
- Streamlit user interface
- Chat-style question and answer experience

## 🧠 RAG Architecture

```text
HR Policy PDF
      ↓
   PyMuPDF
      ↓
 Text Extraction
      ↓
    Chunking
      ↓
Sentence Transformers
      ↓
   Embeddings
      ↓
     FAISS
      ↓
Relevant Policy Chunks
      ↓
    Groq API
      ↓
   HR Answer
```

## 🛠️ Technologies

- Python
- Streamlit
- FAISS
- Sentence Transformers
- PyMuPDF
- Groq API
- NumPy

## 📄 How to Use

1. Open the application.
2. Upload an HR policy PDF.
3. Wait for the document to be processed.
4. Ask a question about the policy.
5. The application searches the relevant document chunks.
6. Groq generates an answer using the retrieved context.
7. Relevant policy pages are displayed as sources.

## 💡 Example Questions

```text
How many annual leave days are employees entitled to?

What is the company's sick leave policy?

What happens if an employee arrives late?

How does the maternity leave policy work?

What is the policy for remote work?

What are the working hours?

What is the resignation notice period?
```

## ⚠️ Important

The assistant is designed to answer questions based on the uploaded HR policy.

If information is not available in the retrieved policy context, the application instructs the model not to invent an answer.

This application should not be treated as legal advice or as a replacement for an organization's HR department.


## 📌 Future Improvements

Possible future features include:

- Multiple PDF support
- Persistent FAISS indexes
- Document history
- Better citation display
- Page previews
- Authentication
- Conversation export
- Multiple HR policy documents
- Policy comparison
- Admin dashboard
- More advanced chunking
