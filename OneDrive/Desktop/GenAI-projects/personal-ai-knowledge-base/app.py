import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from google import genai
import os
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personal PDF AI Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("❌ Gemini API key not found.")
    st.stop()

client = genai.Client(api_key=api_key)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    """
    Clean extracted PDF text.
    """

    text = text.replace("\x00", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pdf(uploaded_file):
    """
    Extract text from every page of a PDF.

    Returns:
        list of dictionaries containing page number and text.
    """

    documents = []

    try:

        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                page_text = page.extract_text()
            except Exception:
                page_text = ""

            if page_text:

                page_text = clean_text(page_text)

                if page_text:

                    documents.append(
                        {
                            "page": page_number,
                            "text": page_text
                        }
                    )

    except Exception as error:

        st.error(f"❌ Could not read PDF: {error}")

    return documents


def create_chunks(text, chunk_size=120, overlap=30):
    """
    Split text into overlapping chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_pdf_chunks(pdf_pages, filename):
    """
    Convert PDF pages into searchable chunks.
    """

    all_chunks = []

    for page_data in pdf_pages:

        page_number = page_data["page"]
        text = page_data["text"]

        chunks = create_chunks(text)

        for chunk_number, chunk in enumerate(chunks, start=1):

            all_chunks.append(
                {
                    "filename": filename,
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk
                }
            )

    return all_chunks


def find_relevant_chunks(question, documents, top_k=5):
    """
    Find the most relevant PDF chunks using TF-IDF similarity.
    """

    if not documents:
        return []

    texts = [document["text"] for document in documents]

    all_text = [question] + texts

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )

        vectors = vectorizer.fit_transform(all_text)

        question_vector = vectors[0]

        document_vectors = vectors[1:]

        similarities = cosine_similarity(
            question_vector,
            document_vectors
        ).flatten()

        ranked_indexes = similarities.argsort()[::-1]

        results = []

        for index in ranked_indexes[:top_k]:

            score = float(similarities[index])

            if score > 0:

                document = documents[index].copy()

                document["score"] = score

                results.append(document)

        return results

    except Exception:
        return []


def create_context(relevant_chunks):
    """
    Create context for Gemini.
    """

    context_parts = []

    for item in relevant_chunks:

        context_parts.append(
            f"""
SOURCE FILE: {item['filename']}
PAGE: {item['page']}

CONTENT:
{item['text']}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="text-align:center;">
            📚 PDF Knowledge Base
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.write("Upload PDFs and ask questions about them.")

    st.divider()

    uploaded_files = st.file_uploader(
        "📄 Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    if uploaded_files:

        st.success(
            f"📚 {len(uploaded_files)} PDF(s) uploaded"
        )

        if st.button(
            "🔄 Process PDFs",
            use_container_width=True
        ):

            with st.spinner("📖 Reading your PDFs..."):

                all_documents = []
                pdf_names = []

                for uploaded_file in uploaded_files:

                    pdf_pages = extract_pdf(uploaded_file)

                    if not pdf_pages:
                        st.warning(
                            f"⚠️ No readable text found in {uploaded_file.name}"
                        )
                        continue

                    chunks = build_pdf_chunks(
                        pdf_pages,
                        uploaded_file.name
                    )

                    all_documents.extend(chunks)

                    pdf_names.append(uploaded_file.name)

                st.session_state.documents = all_documents
                st.session_state.pdf_names = pdf_names
                st.session_state.messages = []

            if all_documents:

                st.success(
                    f"✅ PDFs processed successfully!"
                )

                st.info(
                    f"🧩 {len(all_documents)} text chunks created"
                )

            else:

                st.error(
                    "❌ No readable text was found in the uploaded PDFs."
                )

    if st.session_state.pdf_names:

        st.divider()

        st.markdown("### 📄 Uploaded PDFs")

        for name in st.session_state.pdf_names:

            st.write(f"📄 {name}")

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown("### 💡 How to use")

    st.caption(
        """
        1. Upload a PDF  
        2. Click Process PDFs  
        3. Ask a question  
        4. Get an answer from your PDF
        """
    )

    st.divider()

    st.caption("🤖 Powered by Google Gemini")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📚 Personal PDF AI Assistant
    </div>

    <div class="subtitle">
        Upload your PDF and ask questions about its content.
    </div>
    """,
    unsafe_allow_html=True
)


st.info(
    "💡 Upload one or more PDFs, process them, and ask questions. "
    "Answers are generated using the information found in your PDFs."
)


# ============================================================
# SHOW STATUS
# ============================================================

if st.session_state.documents:

    st.success(
        f"✅ Ready! "
        f"{len(st.session_state.documents)} searchable chunks available."
    )

else:

    st.warning(
        "📄 Please upload a PDF from the sidebar and click "
        "'Process PDFs' before asking questions."
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if "sources" in message:

            for source in message["sources"]:

                st.caption(
                    f"📄 {source['filename']} — Page {source['page']}"
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your uploaded PDF..."
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if question:

    if not st.session_state.documents:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.write(question)


    # --------------------------------------------------------
    # RETRIEVE RELEVANT PDF CONTENT
    # --------------------------------------------------------

    relevant_chunks = find_relevant_chunks(
        question,
        st.session_state.documents,
        top_k=5
    )


    if not relevant_chunks:

        answer = (
            "I couldn't find information related to your question "
            "in the uploaded PDF."
        )

        with st.chat_message("assistant"):

            st.write(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = create_context(
        relevant_chunks
    )


    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a PDF question-answering assistant.

Your job is to answer the user's question ONLY using the
information provided in the PDF context below.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

IMPORTANT RULES:

1. Answer only from the PDF context.
2. Do not use outside knowledge.
3. Do not invent information.
4. If the answer is not available in the PDF context,
   clearly say:

   "The information is not available in the uploaded PDF."

5. Give a clear and simple answer.
6. If possible, explain the answer in a few sentences.
"""


    # --------------------------------------------------------
    # GEMINI RESPONSE
    # --------------------------------------------------------

    try:

        with st.chat_message("assistant"):

            with st.spinner("🤖 Reading your PDF and thinking..."):

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                answer = response.text

            st.write(answer)

            st.markdown("**📚 Sources used:**")

            displayed_sources = []

            for item in relevant_chunks:

                source_key = (
                    item["filename"],
                    item["page"]
                )

                if source_key not in displayed_sources:

                    displayed_sources.append(source_key)

                    st.caption(
                        f"📄 {item['filename']} — Page {item['page']}"
                    )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [
                    {
                        "filename": item["filename"],
                        "page": item["page"]
                    }
                    for item in relevant_chunks
                ]
            }
        )


    except Exception as error:

        st.error(
            f"❌ Gemini Error: {error}"
        )