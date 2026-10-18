import re
import streamlit as st
import pymupdf as fitz
import os
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Name of the ChromaDB collection used to store PDF chunks
COLLECTION_NAME = "pdf_documents"

# 2. Load the embedding model (cached so it is not reloaded on every rerun)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3. Create a persistent ChromaDB database
@st.cache_resource
def load_database():
    return chromadb.PersistentClient(path="./chroma_db")

# Check whether the PDF collection already exists in ChromaDB
def get_existing_collection(client, collection_name=COLLECTION_NAME):
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return None

# Clean extracted PDF text by removing invisible characters and fixing spacing
def clean_pdf_text(text: str) -> str:
    if not text:
        return ""

    text = (
        text.replace("\x00", "")
        .replace("\x0c", "\n")
        .replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    text = text.replace("\uf0b7", "•").replace("\uf0a7", "•").replace("\uf0d8", "•")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"(\b[A-Za-z]+)-\s*\n\s*([a-z]+)\b", r"\1\2", text)

    text = re.sub(r"(\b[A-Za-z]+)-\s+([A-Za-z]+\b)", r"\1-\2", text)

    text = re.sub(r"[^\S\n]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()

# 4. Split text into overlapping chunks while preserving each page number
def create_page_chunks(page_texts, chunk_size=100, overlap=20):
    chunks = []

    for page_number, page_text in enumerate(page_texts, start=1):
        words = page_text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page_number": page_number
                })

            if end >= len(words):
                break

            start = end - overlap

    return chunks

# 8. Retrieve the most relevant chunks for a question using semantic search
def retrieve_chunks(question, top_k=3, db_client=None):
    client = db_client if db_client is not None else load_database()
    collection = get_existing_collection(client, COLLECTION_NAME)
    if collection is None:
        raise ValueError("Document collection not found. Please upload and index a PDF first.")

    total_chunks = collection.count()
    if total_chunks == 0:
        raise ValueError("The document collection is empty. Please upload and index a PDF first.")

    # Convert the question into an embedding vector
    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    # Search ChromaDB for the closest matching chunks
    n_results = min(top_k, total_chunks)
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    return results

# 9. Generate an answer with Groq using only the retrieved context
def generate_answer(question, context):
    prompt = f"""
    You are a helpful assistant answering questions about a PDF study material.

    Answer using only the context provided below.
    If the answer is not present in the context,
    say that you could not find the answer in the study material.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it in Streamlit app Settings → Secrets.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

# UI styling: keep the Learnly AI design separate from the RAG logic
def inject_learnly_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background: linear-gradient(135deg, #f7f5fd 0%, #faf9fe 50%, #f4f0fc 100%) !important;
            color: #18181b !important;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 840px !important;
            padding: 2.5rem 3rem 4rem 3rem !important;
            background: #ffffff !important;
            border-radius: 28px !important;
            border: 1px solid #ebe6f7 !important;
            box-shadow: 0 16px 48px rgba(139, 92, 246, 0.05), 0 2px 8px rgba(0, 0, 0, 0.02) !important;
            margin: 2.5rem auto !important;
        }

        .learnly-orb-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding-top: 6px;
            padding-bottom: 12px;
        }
        .learnly-orb {
            width: 76px;
            height: 76px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 30%, #ffffff 0%, #f3e8ff 22%, #d8b4fe 45%, #a855f7 75%, #7e22ce 100%);
            box-shadow: 0 0 42px rgba(168, 85, 247, 0.38),
                        0 0 14px rgba(192, 132, 252, 0.55),
                        inset 0 0 18px rgba(255, 255, 255, 0.7);
            filter: drop-shadow(0 8px 20px rgba(147, 51, 234, 0.25));
            animation: orb-pulse 4s ease-in-out infinite alternate;
        }
        @keyframes orb-pulse {
            0% { transform: translateY(0px) scale(1); box-shadow: 0 0 40px rgba(168, 85, 247, 0.38); }
            100% { transform: translateY(-5px) scale(1.02); box-shadow: 0 0 54px rgba(168, 85, 247, 0.54); }
        }

        .learnly-header {
            text-align: center;
            margin-bottom: 4px;
        }
        .learnly-title {
            color: #111827;
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 4px;
        }
        .learnly-tagline {
            color: #7c3aed;
            font-size: 1.05rem;
            font-weight: 500;
            letter-spacing: -0.01em;
            margin-bottom: 28px;
        }
        .learnly-description {
            max-width: 620px;
            margin: 0 auto 30px auto;
            color: #6b7280;
            font-size: 1rem;
            line-height: 1.7;
            font-weight: 400;
            text-align: center;
        }

        .learnly-card {
            background: #ffffff;
            border: 1px solid #ede8f7;
            border-radius: 20px;
            padding: 22px 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 18px rgba(139, 92, 246, 0.03);
        }

        button[kind="primary"] {
            background: #8b5cf6 !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.25) !important;
            transition: all 0.2s ease !important;
        }
        button[kind="primary"]:hover {
            background: #7c3aed !important;
            box-shadow: 0 6px 18px rgba(124, 58, 237, 0.35) !important;
            transform: translateY(-1px);
        }

        div[data-baseweb="input"] {
            border-radius: 14px !important;
            border: 1.5px solid #ebe5f7 !important;
            background: #faf9fd !important;
            transition: all 0.2s ease !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #8b5cf6 !important;
            background: #ffffff !important;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
        }

        div[data-testid="stExpander"] {
            border-radius: 14px !important;
            border: 1px solid #ede8f7 !important;
            background: #faf9fd !important;
            margin-bottom: 10px !important;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: #1f2937 !important;
            padding: 10px 14px !important;
        }
        div[data-testid="stExpander"] summary:hover {
            color: #7c3aed !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] small,
        section[data-testid="stFileUploaderDropzone"] small {
            font-size: 0 !important;
            visibility: hidden !important;
            position: relative !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

# 10. Streamlit application: upload a PDF, ask questions, and show answers
def main():
    st.set_page_config(
        page_title="Learnly AI",
        page_icon="📖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Apply the existing UI theme and connect to the database
    inject_learnly_theme()

    db = load_database()

    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = None
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "generation_error" not in st.session_state:
        st.session_state["generation_error"] = None
    if "indexed_filename" not in st.session_state:
        st.session_state["indexed_filename"] = None

    st.markdown(
        """
        <div class="learnly-orb-container">
            <div class="learnly-orb"></div>
        </div>
        <div class="learnly-header">
            <div class="learnly-title">Learnly AI</div>
            <div class="learnly-tagline">Your study materials. Smarter answers.</div>
            <div class="learnly-description">Upload your notes, textbooks, and study PDFs. Ask questions and get AI-generated answers grounded in your materials, with page references.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5. Upload a PDF and extract its text page by page
    st.markdown("### 📄 Upload Study Materials")
    uploaded_file = st.file_uploader(
        "Upload Study Materials",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload your study materials in PDF format."
    )

    if uploaded_file is not None:
        file_col, action_col = st.columns([3, 1])
        with file_col:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:8px; padding:8px 12px; background:#faf9fe; border:1px solid #ede8f7; border-radius:10px; font-size:0.9rem; color:#374151;">
                    <span>📎</span> <strong>{uploaded_file.name}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
        with action_col:
            process_clicked = st.button("Process PDF", type="primary", use_container_width=True)

        if process_clicked:
            try:
                with st.spinner("Processing study materials..."):
                    pdf_bytes = uploaded_file.getvalue()

                    # Extract text from every page in the uploaded PDF
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                        raw_page_texts = [page.get_text("text") for page in document]

                    # Clean the extracted text before splitting it into chunks
                    cleaned_page_texts = [clean_pdf_text(t) for t in raw_page_texts]
                    full_text = "\n".join(cleaned_page_texts)

                    if not full_text.strip():
                        st.warning("⚠️ No readable text found in this PDF. It may be scanned or empty.")
                    else:
                        # 6. Create chunks and keep their original page numbers
                        chunks = create_page_chunks(cleaned_page_texts, chunk_size=100, overlap=20)
                        chunk_texts = [chunk["text"] for chunk in chunks]

                        # 7. Convert each chunk into an embedding vector
                        model = load_embedding_model()
                        embeddings = model.encode(
                            chunk_texts,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        ).tolist()

                        # Replace the previous indexed PDF with this PDF
                        try:
                            db.delete_collection(COLLECTION_NAME)
                        except Exception:
                            pass

                        collection = db.create_collection(
                            name=COLLECTION_NAME,
                            metadata={"hnsw:space": "cosine"}
                        )

                        # Store chunk text, embeddings, and source/page metadata
                        collection.add(
                            ids=[f"chunk_{i}" for i in range(len(chunks))],
                            documents=chunk_texts,
                            embeddings=embeddings,
                            metadatas=[
                                {
                                    "source": uploaded_file.name,
                                    "chunk_number": i + 1,
                                    "page_number": chunk["page_number"]
                                }
                                for i, chunk in enumerate(chunks)
                            ]
                        )

                        st.session_state["indexed_filename"] = uploaded_file.name
                        st.success(f"✅ Study materials processed and ready!")
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")

    elif st.session_state.get("indexed_filename"):
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:8px; padding:6px 10px; background:#f5f3ff; border:1px solid #ddd6fe; border-radius:10px; font-size:0.85rem; color:#6d28d9; margin-bottom:12px;">
                <span>✓</span> Indexed study material: <strong>{st.session_state['indexed_filename']}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 8. Accept a question and retrieve relevant chunks from the indexed PDF
    st.markdown("### 💬 Ask a Question")
    top_k = st.selectbox(
    "Retrieved passages",
    options=[1, 2, 3, 4, 5, 6, 8, 10],
    index=2,
    help="Choose how many relevant passages Learnly AI should retrieve."
    )
    with st.form("learnly_query_form", clear_on_submit=False):
        question_input = st.text_input(
            "Ask anything about your study material...",
            placeholder="Ask anything about your study material...",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button("Ask Learnly", type="primary", use_container_width=True)

    if submit_button:
        cleaned_question = question_input.strip()
        if not cleaned_question:
            st.warning("⚠️ Please enter a question about your study material.")
        else:
            st.session_state["last_answer"] = None
            st.session_state["last_results"] = None
            st.session_state["generation_error"] = None

            collection = get_existing_collection(db, COLLECTION_NAME)
            if collection is None:
                st.info("ℹ️ Please upload study material and click **Process PDF** first.")
            elif collection.count() == 0:
                st.info("ℹ️ The study collection is currently empty. Please upload and process a PDF.")
            else:
                try:
                    with st.spinner("Finding relevant concepts in your study material..."):
                        results = retrieve_chunks(cleaned_question, top_k=top_k, db_client=db)
                        st.session_state["last_results"] = results

                    if (
                        not results
                        or not results.get("documents")
                        or len(results["documents"]) == 0
                        or len(results["documents"][0]) == 0
                    ):
                        st.warning("⚠️ No relevant concepts were found in the study material for this question.")
                    else:
                        # Combine retrieved chunks into the context for the language model
                        context = "\n\n".join(results["documents"][0])
                        with st.spinner("Generating answer..."):
                            try:
                                answer = generate_answer(cleaned_question, context)
                                st.session_state["last_answer"] = answer
                            except Exception as ollama_err:
                                st.session_state["generation_error"] = str(ollama_err)

                except Exception as retrieval_error:
                    st.error(f"❌ Search failed: {retrieval_error}")

    # 11. Display the generated answer and the source passages with page numbers
    if st.session_state.get("last_answer"):
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-top:28px; margin-bottom:8px;">
                <span style="font-size:1.15rem; font-weight:700; color:#111827;">💡 Answer</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.container(border=True):
            st.markdown(st.session_state["last_answer"])
        st.write("")

    elif st.session_state.get("generation_error"):
        err_str = st.session_state["generation_error"]
        st.markdown("### 💡 Answer")
        st.error(f"⚠️ **Answer generation failed:** {err_str}")

    results = st.session_state.get("last_results")
    if results and results.get("documents") and len(results["documents"]) > 0:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)

        if len(documents) > 0:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:8px; margin-top:22px; margin-bottom:10px;">
                    <span style="font-size:1.05rem; font-weight:700; color:#111827;">📚 Source References</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            for idx, (doc_text, metadata) in enumerate(zip(documents, metadatas), start=1):
                source_file = metadata.get("source", "Study Material")
                page_num = metadata.get("page_number", "N/A")

                expander_title = f"📄 {source_file} — Page {page_num}"

                with st.expander(expander_title, expanded=(idx == 1)):
                    st.write(doc_text)

if __name__ == "__main__":
    main()
