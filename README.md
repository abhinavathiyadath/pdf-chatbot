# Learnly AI

**Your study materials. Smarter answers.**

Learnly AI is an AI-powered PDF study assistant that lets you upload study materials, ask questions, and receive answers grounded in retrieved passages, with source page references.

Built using **Retrieval-Augmented Generation (RAG)**, Learnly AI extracts and cleans PDF text, retrieves relevant passages through semantic search, and uses the Groq API to generate answers based on your study material.

## ✨ Features

- **PDF text extraction:** Extracts text from PDF documents using PyMuPDF.
- **Text cleaning:** Normalizes whitespace, removes invisible characters, standardizes bullet symbols, and repairs some words split across line breaks.
- **Page-aware chunking:** Splits text into chunks of 100 words with a 20-word overlap while preserving page numbers.
- **Semantic search:** Uses Sentence Transformers to retrieve relevant passages.
- **Vector storage:** Stores embeddings, text chunks, and metadata in ChromaDB.
- **Configurable retrieval:** Lets users select the number of relevant passages retrieved.
- **Source references:** Displays supporting passages with filenames and page numbers.
- **AI-powered answers:** Uses the Groq API to generate answers grounded in retrieved PDF context.
- **Simple interface:** Provides a Streamlit UI for uploading PDFs and asking questions.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | User interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| ChromaDB | Vector database |
| Groq API | Language-model inference |

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
Generate Answer (Groq API)
    |
    v
Display Answer and Sources
```

### RAG Pipeline

1. **Upload:** The user uploads a PDF study document.
2. **Extract:** PyMuPDF extracts text from each page.
3. **Clean:** Text is normalized to reduce common PDF formatting artifacts.
4. **Chunk:** Text is divided into smaller passages while preserving page references.
5. **Embed:** Sentence Transformers converts each passage into a numerical vector.
6. **Store:** ChromaDB stores passages, embeddings, and metadata.
7. **Retrieve:** The question is embedded and compared with stored passages using cosine distance.
8. **Generate:** The retrieved context is sent to the Groq API to generate an answer.
9. **Display:** Learnly AI presents the answer and supporting source passages.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- A Groq API key
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd <your-project-folder>
```

Replace the placeholders with your actual repository URL and folder name.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bat
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Make sure `requirements.txt` includes the packages required by your application, including `groq`.

### 4. Configure Your Groq API Key

Create a local Streamlit secrets file:

```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

Add the following to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Replace the placeholder with your actual API key. Never commit your API key or secrets file to GitHub.

For Streamlit Community Cloud, add the same `GROQ_API_KEY` setting under your app's **Settings → Secrets**.

### 5. Run Learnly AI

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

Upload a PDF, click **Process PDF**, and ask a question about its contents.

## 📁 Project Structure

```text
Learnly-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml
└── chroma_db/
```

The `.streamlit/secrets.toml` file contains your local API key and must not be committed. The `chroma_db` directory is created at runtime.

## 🔒 Privacy and API Usage

- PDF text extraction and embedding generation run in the application's runtime environment.
- ChromaDB stores document vectors and metadata in the application's runtime environment.
- Answer generation uses the Groq API, so retrieved context is sent to an external provider.
- A valid Groq API key and internet connection are required for answer generation.

This version is not fully local or offline. Avoid submitting sensitive documents unless you are comfortable with the provider's data-handling policies.

## 🎨 UI Development

AI-assisted development tools were used to help refine the user interface, including styling and layout. The project also involves integrating the interface with the PDF processing and RAG pipeline, testing functionality, and understanding the underlying implementation.

## 🔮 Future Improvements

- Support indexing multiple PDF documents simultaneously.
- Improve retrieval quality with hybrid search or reranking.
- Add automated tests for PDF cleaning, chunking, retrieval, and metadata preservation.
- Add OCR support for scanned PDFs.
- Support conversation context for follow-up questions.
- Evaluate answer quality and citation accuracy across different study materials.

## ⚠️ Limitations

- Scanned PDFs may not contain extractable text and may require OCR.
- Answer quality depends on PDF text quality, retrieved passages, and the language model.
- Source page references identify where passages originated; they do not guarantee that every generated statement is fully supported.
- The current implementation replaces the existing indexed collection when a new PDF is processed, so simultaneous multi-document support is not included.
- Hosted inference requires internet access and a valid Groq API key.

## 👨‍💻 Author

**Abhinav Athiyadath**

Learnly AI is a project exploring Retrieval-Augmented Generation, semantic search, vector databases, and AI-assisted study tools.
