
import streamlit as st
import fitz
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="My RAG Project",
    page_icon="📚"
)

st.title("📚 My First RAG Project")
st.write("Extract, chunk, embed, and store PDF text.")


# 1. Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# 2. Create a persistent ChromaDB database
@st.cache_resource
def load_database():
    return chromadb.PersistentClient(path="./chroma_db")


db = load_database()

# Step 7: Retrieve relevant chunks
def retrieve_chunks(question, top_k=3):
    # Load the embedding model
    model = load_embedding_model()

    # Convert the question into a vector
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    # Access the stored PDF chunks
    collection = db.get_collection("pdf_documents")

    # Search for the most relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results



# 3. Split text into chunks
def split_into_chunks(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# 4. Upload PDF
uploaded_file = st.file_uploader(
    "Choose your PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    # 5. Extract text page by page
    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:
        page_texts = [
            page.get_text("text")
            for page in document
        ]
        total_pages = len(document)

    full_text = "\n".join(page_texts)

    if not full_text.strip():
        st.warning("No readable text found in this PDF.")

    else:
        st.success("PDF text extracted successfully!")
        st.write("Total pages:", total_pages)

        # 6. Create chunks
        chunks = split_into_chunks(full_text)

        st.write("Total words:", len(full_text.split()))
        st.write("Total chunks:", len(chunks))

        # 7. Generate embeddings and store in ChromaDB
        if st.button("Store in ChromaDB", type="primary"):
            try:
                with st.spinner(
                    "Generating embeddings and storing chunks..."
                ):
                    model = load_embedding_model()

                    embeddings = model.encode(
                        chunks,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    ).tolist()

                    # Create a collection for this PDF
                    collection_name = "pdf_documents"

                    # Replace the previous collection on re-indexing
                    try:
                        db.delete_collection(collection_name)
                    except Exception:
                        pass

                    collection = db.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )

                    # Store chunks, embeddings and metadata
                    collection.add(
                        ids=[
                            f"chunk_{i}"
                            for i in range(len(chunks))
                        ],
                        documents=chunks,
                        embeddings=embeddings,
                        metadatas=[
                            {
                                "source": uploaded_file.name,
                                "chunk_number": i + 1
                            }
                            for i in range(len(chunks))
                        ]
                    )

                st.session_state["indexed"] = True
                st.session_state["indexed_filename"] = (
                    uploaded_file.name
                )

                st.success(
                    f"Successfully stored {len(chunks)} chunks!"
                )

            except Exception as error:
                st.error(f"Indexing failed: {error}")

        # 8. Show stored data
        if (
            st.session_state.get("indexed")
            and st.session_state.get("indexed_filename")
            == uploaded_file.name
        ):
            try:
                collection = db.get_collection("pdf_documents")

                st.subheader("ChromaDB Storage")
                st.write("Stored chunks:", collection.count())

                if collection.count() > 0:
                    stored_data = collection.get(
                        limit=1,
                        include=["documents", "metadatas"]
                    )

                    st.write("Example stored chunk:")
                    st.write(stored_data["documents"][0])

                    st.write("Metadata:")
                    st.json(stored_data["metadatas"][0])

                    
                    # Step 8: Search the stored PDF
                    st.subheader("Ask a question about your PDF")

                    question = st.text_input(
                        "Enter your question",
                        key="retrieval_question"
                    )

                    if st.button("Find relevant passages"):
                        if question.strip():
                            try:
                                results = retrieve_chunks(question)

                                st.success("Retrieved relevant passages!")

                                for i, text in enumerate(
                                    results["documents"][0],
                                    start=1
                                ):
                                    metadata = results["metadatas"][0][i - 1]
                                    distance = results["distances"][0][i - 1]

                                    st.markdown(f"### Result {i}")
                                    st.write(text)
                                    st.caption(
                                        f"Source: {metadata['source']} | "
                                        f"Chunk: {metadata['chunk_number']} | "
                                        f"Cosine distance: {distance:.4f}"
                                    )

                            except Exception as error:
                                st.error(f"Retrieval failed: {error}")
                        else:
                            st.warning("Please enter a question.")


            except Exception as error:
                st.error(f"Could not read stored data: {error}")

