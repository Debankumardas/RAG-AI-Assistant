# 🤖 RAG AI Assistant

A document-based AI assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions from uploaded PDF documents.

## 📌 Overview

The **RAG AI Assistant** allows users to upload a PDF document and ask questions about its content through a conversational interface.

Instead of sending the entire document directly to the AI model, the system:

1. Extracts text from the uploaded PDF.
2. Splits the text into smaller chunks.
3. Converts the chunks into vector representations.
4. Stores the vectors for similarity-based retrieval.
5. Retrieves the most relevant document content for a user's question.
6. Sends the retrieved context to Google's Gemini model.
7. Generates an answer based on the retrieved document information.

This approach helps the assistant provide answers grounded in the uploaded document.

---

## ✨ Features

* 📄 PDF document upload
* 📝 Automatic PDF text extraction
* ✂️ Text chunking
* 🧠 Sentence Transformer embeddings
* 🔎 FAISS similarity search
* 📚 Retrieval-Augmented Generation (RAG)
* 🤖 Google Gemini-powered responses
* 💬 Chat-style Gradio interface
* 🗑️ Clear chat functionality
* 📊 Document processing information
* 🔐 Environment-based API key configuration

---

## 🛠️ Technologies Used

| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| **Python**                | Core programming language       |
| **Gradio**                | Web-based user interface        |
| **Google Gemini**         | AI response generation          |
| **Google GenAI SDK**      | Gemini API integration          |
| **PyMuPDF**               | PDF text extraction             |
| **Sentence Transformers** | Text embeddings                 |
| **FAISS**                 | Vector similarity search        |
| **NumPy**                 | Numerical operations            |
| **python-dotenv**         | Environment variable management |

---

## 🏗️ System Architecture

```text
                    ┌─────────────────┐
                    │    User Uploads │
                    │       PDF       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PDF Processing │
                    │    PyMuPDF      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text Chunking   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Embeddings    │
                    │Sentence         │
                    │Transformers     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ FAISS Vector    │
                    │     Store       │
                    └────────┬────────┘
                             │
                 User Question
                             │
                             ▼
                    ┌─────────────────┐
                    │   Retrieval     │
                    │ Relevant Chunks │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Google Gemini   │
                    │      LLM        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Final Answer    │
                    └─────────────────┘
```

---

## 🔄 How RAG Works in This Project

The project follows the following pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embedding Generation
 ↓
FAISS Vector Store
 ↓
User Question
 ↓
Similarity Retrieval
 ↓
Relevant Document Context
 ↓
Gemini
 ↓
Generated Answer
```

The key idea is that **Gemini receives relevant information retrieved from the uploaded document**, allowing the response to be grounded in the document rather than relying only on the model's general knowledge.

---

## 📂 Project Structure

```text
RAG-AI-Assistant/
│
├── app.py
├── config.py
├── rag_engine.py
├── document_manager.py
├── document_processor.py
├── create_vectorstore.py
├── check_models.py
├── test_retrieval.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── sample.pdf
```

### Main Files

**`app.py`**

Contains the Gradio user interface and application interaction logic.

**`rag_engine.py`**

Handles the RAG question-answering process and interaction with the language model.

**`document_manager.py`**

Manages document processing and vector-store related operations.

**`document_processor.py`**

Handles PDF text extraction and document chunking.

**`create_vectorstore.py`**

Creates the vector-store data used for document retrieval.

**`test_retrieval.py`**

Used to test whether relevant document chunks can be retrieved for a query.

**`config.py`**

Handles application configuration and environment-based settings.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Debankumardas/RAG-AI-Assistant.git
```

Then enter the project directory:

```bash
cd RAG-AI-Assistant
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Configuration

The project uses the Google Gemini API.

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

**Never upload the `.env` file or your API key to GitHub.**

The project uses `.gitignore` to keep sensitive configuration files out of version control.

---

## ▶️ Running the Application

Start the application using:

```bash
python app.py
```

The Gradio interface will open in your browser.

---

## 📄 Using the Application

### Step 1 — Start the application

```bash
python app.py
```

### Step 2 — Upload a PDF

Use the PDF upload option in the application.

### Step 3 — Wait for processing

The application processes the document and prepares it for retrieval.

### Step 4 — Ask a question

For example:

```text
What is the minimum attendance requirement?
```

### Step 5 — Receive the answer

The system retrieves relevant content from the document and uses Gemini to generate the response.

---

## 🧪 Example

### Question

```text
What is the minimum attendance requirement?
```

### Answer

```text
The minimum attendance requirement is 75 percent in each course.
```

The answer is generated using information retrieved from the uploaded document.

---

## 🎯 Why Use RAG?

A standard language model does not automatically know the contents of a user's private documents.

RAG provides a solution by retrieving relevant information from the document and supplying that information to the language model as context.

This allows the system to answer questions based on the uploaded document.

### Traditional LLM

```text
User Question
     ↓
LLM
     ↓
General Knowledge
     ↓
Answer
```

### RAG System

```text
User Question
     ↓
Document Retrieval
     ↓
Relevant Context
     ↓
Gemini
     ↓
Document-Based Answer
```

---

## 🔐 Security

The following files should **never** be uploaded to GitHub:

```text
.env
venv/
__pycache__/
*.pyc
```

The Gemini API key must remain private.

---

## ⚠️ Limitations

* Answer quality depends on the quality and structure of the uploaded PDF.
* Scanned or image-only PDFs may require OCR for reliable text extraction.
* The system requires a valid Gemini API key.
* Very large documents may require additional optimization.
* Retrieval quality depends on the document chunking and embedding process.
* The system is designed primarily for document-based question answering.

---

## 🚀 Future Improvements

Possible future improvements include:

* Multiple-document support
* Improved conversation memory
* Page-level source citations
* OCR support for scanned PDFs
* Improved retrieval and reranking
* Persistent document management
* Better handling of large documents
* Cloud deployment

---

## 📚 Project Purpose

This project demonstrates the practical implementation of a **Retrieval-Augmented Generation system** by combining:

**Document Processing + Embeddings + Vector Search + Large Language Models**

It provides a practical example of how AI systems can retrieve information from user-provided documents and generate context-aware responses.

---

## 👨‍💻 Author

**Deban Kumar Das**

BCA — Data Science

---

## ⭐ Acknowledgement

This project was developed as a learning and portfolio project to explore **RAG, vector databases, document processing, embeddings, and generative AI applications**.
