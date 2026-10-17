
import streamlit as st
import fitz

st.set_page_config(
    page_title="My RAG Project",
    page_icon="📚"
)

st.title("📚 My First RAG Project")
st.write("Upload a PDF, extract text, and split it into chunks.")

# Step 1: Upload PDF
uploaded_file = st.file_uploader(
    "Choose your PDF",
    type=["pdf"]
)

# Step 2: Split extracted text into chunks
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


# Step 3: Extract PDF text
if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        full_text = ""

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            full_text += (
                f"\n--- Page {page_number} ---\n{page_text}"
            )

        st.success("PDF uploaded successfully!")
        st.write("Total pages:", len(document))

    # Step 4: Create chunks
    chunks = split_into_chunks(
        full_text,
        chunk_size=100,
        overlap=20
    )

    st.subheader("Text Chunking Results")
    st.write("Total words:", len(full_text.split()))
    st.write("Total chunks:", len(chunks))

    # Step 5: Preview chunks
    for index, chunk in enumerate(chunks):
        with st.expander(f"Chunk {index + 1}"):
            st.write(chunk)
