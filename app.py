
import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="My RAG Project",
    page_icon="📚"
)

st.title("📚 My First RAG Project")
st.write("Upload a PDF, split its text, and generate embeddings.")


# Step 1: Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Step 2: Split text into chunks
def split_into_chunks(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# Step 3: Upload PDF
uploaded_file = st.file_uploader(
    "Choose your PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    # Step 4: Extract text
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        page_texts = [
            page.get_text("text")
            for page in document
        ]
        total_pages = len(document)

    full_text = "\n".join(page_texts)

    if not full_text.strip():
        st.warning(
            "No readable text found. Try a text-based PDF."
        )
    else:
        st.success("PDF text extracted successfully!")
        st.write("Total pages:", total_pages)

        # Step 5: Create chunks
        chunks = split_into_chunks(
            full_text,
            chunk_size=100,
            overlap=20
        )

        st.subheader("Chunking Results")
        st.write("Total words:", len(full_text.split()))
        st.write("Total chunks:", len(chunks))

        # Step 6: Generate embeddings when requested
        if st.button("Generate Embeddings", type="primary"):
            with st.spinner("Loading model and generating embeddings..."):
                model = load_embedding_model()

                embeddings = model.encode(
                    chunks,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

            st.success("Embeddings generated successfully!")

            st.write("Number of chunks embedded:", len(embeddings))
            st.write("Dimensions per embedding:", embeddings.shape[1])

            # Display one example
            st.subheader("Embedding Preview")
            st.write("First chunk:")
            st.write(chunks[0])

            st.write("First 10 numbers in its embedding:")
            st.code(str(embeddings[0][:10].tolist()))

            # Store results for display during subsequent reruns
            st.session_state["chunks"] = chunks
            st.session_state["embeddings"] = embeddings.tolist()

        # Display chunk previews
        with st.expander("View text chunks"):
            for index, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {index + 1}**")
                st.write(chunk)
