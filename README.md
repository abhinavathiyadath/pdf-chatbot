# Learnly AI

> **Your study materials. Smarter answers.**

Learnly AI is a local, AI-powered PDF study assistant that lets you upload study materials, ask questions, and receive context-based answers with source page references.

Built using **Retrieval-Augmented Generation (RAG)**, Learnly AI retrieves relevant passages from your documents and uses a locally running language model to generate answers.

## ✨ Features

- **Local AI inference:** Uses Ollama with Llama 3.2 to generate answers locally.
- **PDF text extraction:** Extracts text from PDF documents using PyMuPDF.
- **Text cleaning:** Normalizes whitespace, removes invisible characters, standardizes common bullet symbols, and repairs words split across line breaks.
- **Page-aware chunking:** Splits extracted text into chunks of 100 words with a 20-word overlap while preserving page numbers.
- **Semantic search:** Uses Sentence Transformers to create embeddings and retrieve relevant passages.
- **Vector storage:** Stores embeddings and document metadata in ChromaDB.
- **Configurable retrieval:** Allows users to select the number of relevant passages retrieved for a question.
- **Source references:** Displays retrieved passages with filename, page number, chunk number, and cosine distance.
- **Error handling:** Handles empty questions, missing document collections, and common Ollama connection or model errors.
- **Simple interface:** Provides a clean Streamlit UI for uploading PDFs and asking questions.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | User interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| ChromaDB | Vector database |
| Ollama | Local LLM inference |
| Llama 3.2 | Answer generation |

## 🧠 How It Works

```text
PDF Upload
    |
    v
Text Extraction (PyMuPDF)
    |
    v
Text Cleaning
    |
    v
Page-Aware Chunking
(100 words, 20-word overlap)
    |
    v
Embedding Generation
(Sentence Transformers)
    |
    v
Store Chunks and Metadata
(ChromaDB)
    |
    v
User Question
    |
    v
Semantic Search
    |
    v
Retrieve Top-K Passages
    |
    v
Generate Answer (Ollama)
    |
    v
Display Answer and Sources
```

### RAG Pipeline

1. **Upload:** The user uploads a PDF study document.
2. **Extract:** PyMuPDF extracts text from each page.
3. **Clean:** Extracted text is normalized to reduce common PDF formatting artifacts.
4. **Chunk:** Text is divided into smaller passages while preserving page references.
5. **Embed:** Sentence Transformers converts each passage into a numerical vector.
6. **Store:** ChromaDB stores the passages, embeddings, and metadata.
7. **Retrieve:** The user's question is embedded and compared with stored passages using cosine distance.
8. **Generate:** Ollama runs Llama 3.2 to produce an answer grounded in the retrieved context.
9. **Display:** Learnly AI presents the answer and supporting passages with source information.

## 🚀 Getting Started

### Prerequisites

Install the following before running the application:

- Python 3.10 or later
- Ollama
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd <your-project-folder>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment using:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure your `requirements.txt` contains the dependencies used by the application.

### 4. Install and Run Ollama

Download Ollama from [ollama.com](https://ollama.com/).

Download the model:

```bash
ollama pull llama3.2
```

Make sure the Ollama service is running. If it is not running automatically, start it with:

```bash
ollama serve
```

### 5. Run Learnly AI

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

Upload a PDF, process it, and ask a question about its contents.

## 📁 Project Structure

```text
Learnly-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── chroma_db/          # Local vector database, created at runtime
```

The exact structure may vary depending on your current implementation. If you add a test suite, include its directory here as well.

## 🔒 Privacy

Learnly AI is designed around local document processing and local model inference.

- PDF text extraction and embedding generation run locally.
- ChromaDB stores document vectors and metadata locally.
- Ollama runs the language model locally.
- No external LLM API key is required.

**Note:** Local model inference does not automatically guarantee that every part of the application is offline. External resources, such as remotely hosted fonts, may still require an internet connection.

## 🎨 UI Development

AI-assisted development tools were used to help refine the user interface, including styling and layout. The project also involves integrating the interface with the PDF processing and RAG pipeline, testing functionality, and understanding the underlying implementation.

## 🔮 Future Improvements

- Support indexing multiple PDF documents simultaneously.
- Improve retrieval quality with hybrid search or reranking.
- Add automated tests for PDF cleaning, chunking, retrieval, and metadata preservation.
- Improve handling of scanned PDFs through OCR.
- Add conversation context for follow-up questions.
- Evaluate answer quality and citation accuracy across different study materials.

## ⚠️ Limitations

- Scanned PDFs may not contain extractable text and may require OCR.
- Answer quality depends on the PDF's text quality, retrieved passages, and the language model.
- Source page references identify where retrieved passages originated; they do not guarantee that every generated statement is fully supported.
- The current implementation replaces the existing indexed collection when a new PDF is processed, so multiple-document support is not yet included.

## 👨‍💻 Author

**Abhinav**

Developed as a project exploring Retrieval-Augmented Generation, semantic search, local LLMs, and AI-assisted study tools.
