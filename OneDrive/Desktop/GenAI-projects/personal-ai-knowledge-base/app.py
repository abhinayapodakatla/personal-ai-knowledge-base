import streamlit as st
from pathlib import Path
from difflib import get_close_matches
import re
import os

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Personal AI Knowledge Base",
    page_icon="🤖",
    layout="wide"
)


# =========================================
# GEMINI SETUP
# =========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ Gemini API key not found.")
    st.stop()

client = genai.Client(api_key=api_key)


# =========================================
# KNOWLEDGE BASE
# =========================================

KNOWLEDGE_FOLDER = Path("knowledge")
CHUNK_SIZE = 50

KNOWLEDGE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# FILE UPLOAD
# =========================================

with st.sidebar:

    st.subheader("📤 Upload Knowledge")

    uploaded_file = st.file_uploader(
        "Upload TXT or PDF file",
        type=["txt", "pdf"]
    )

    if uploaded_file is not None:

        uploaded_file_path = (
            KNOWLEDGE_FOLDER / uploaded_file.name
        )

        if uploaded_file.name.lower().endswith(".txt"):

            uploaded_file_path.write_bytes(
                uploaded_file.getbuffer()
            )

            st.success(
                f"✅ {uploaded_file.name} uploaded!"
            )

        elif uploaded_file.name.lower().endswith(".pdf"):

            try:

                pdf_reader = PdfReader(
                    uploaded_file
                )

                pdf_text = ""

                for page in pdf_reader.pages:

                    text = page.extract_text()

                    if text:
                        pdf_text += text + "\n"

                if pdf_text.strip():

                    txt_file_path = (
                        KNOWLEDGE_FOLDER
                        / f"{uploaded_file.name[:-4]}.txt"
                    )

                    txt_file_path.write_text(
                        pdf_text,
                        encoding="utf-8"
                    )

                    st.success(
                        f"✅ {uploaded_file.name} uploaded and processed!"
                    )

                else:

                    st.error(
                        "❌ Could not extract text from this PDF."
                    )

            except Exception as error:

                st.error(
                    f"❌ PDF Error: {error}"
                )


# =========================================
# LOAD KNOWLEDGE FILES
# =========================================

knowledge_files = list(
    KNOWLEDGE_FOLDER.glob("*.txt")
)

if not knowledge_files:

    st.error(
        "❌ No knowledge files found."
    )

    st.stop()


# =========================================
# TOPIC MAPPING
# =========================================

topic_files = {}

for file in knowledge_files:

    topic = file.stem.lower()

    topic = topic.replace(
        "_notes",
        ""
    )

    topic = topic.replace(
        "_",
        " "
    )

    topic_files[topic] = file


# =========================================
# ALIASES
# =========================================

if "ml" in topic_files:

    ml_file = topic_files["ml"]

    topic_files["machine learning"] = ml_file
    topic_files["regression"] = ml_file
    topic_files["classification"] = ml_file
    topic_files["supervised learning"] = ml_file
    topic_files["unsupervised learning"] = ml_file
    topic_files["training"] = ml_file
    topic_files["model"] = ml_file
    topic_files["prediction"] = ml_file


if "deep learning" in topic_files:

    dl_file = topic_files["deep learning"]

    topic_files["neural network"] = dl_file
    topic_files["neural networks"] = dl_file
    topic_files["cnn"] = dl_file
    topic_files["rnn"] = dl_file
    topic_files["deep neural network"] = dl_file


if "data science" in topic_files:

    ds_file = topic_files["data science"]

    topic_files["probability"] = ds_file
    topic_files["probability theory"] = ds_file
    topic_files["statistics"] = ds_file
    topic_files["data analysis"] = ds_file
    topic_files["data visualization"] = ds_file


if "my" in topic_files:

    my_file = topic_files["my"]

    topic_files["generative ai"] = my_file
    topic_files["generative artificial intelligence"] = my_file
    topic_files["llm"] = my_file
    topic_files["llms"] = my_file
    topic_files["large language model"] = my_file
    topic_files["large language models"] = my_file


# =========================================
# STOP WORDS
# =========================================

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an",
    "tell", "me", "about", "explain", "please",
    "can", "you", "give", "some", "information",
    "on", "do", "know", "i", "want", "to",
    "how", "does", "it", "work", "define",
    "describe", "details", "of", "for", "and",
    "in", "my", "your", "where", "why", "when",
    "which", "used", "use", "uses", "its",
    "something"
}


# =========================================
# TEXT CLEANING
# =========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================
# FIND FILE
# =========================================

def find_file(question):

    question_clean = clean_text(
        question
    )

    topics = sorted(
        topic_files.keys(),
        key=len,
        reverse=True
    )

    for topic in topics:

        if topic in question_clean:

            return topic_files[topic]

    question_words = question_clean.split()

    useful_words = [
        word
        for word in question_words
        if word not in STOP_WORDS
        and len(word) >= 3
    ]

    for word in useful_words:

        matches = get_close_matches(
            word,
            topics,
            n=1,
            cutoff=0.75
        )

        if matches:

            return topic_files[
                matches[0]
            ]

    return None


# =========================================
# CREATE CHUNKS
# =========================================

def create_chunks(content):

    words = content.split()

    chunks = []

    for i in range(
        0,
        len(words),
        CHUNK_SIZE
    ):

        chunk = " ".join(
            words[
                i:i + CHUNK_SIZE
            ]
        )

        if chunk.strip():

            chunks.append(
                chunk
            )

    return chunks


# =========================================
# SCORE CHUNKS
# =========================================

def score_chunk(
    question,
    chunk
):

    question_clean = clean_text(
        question
    )

    chunk_clean = clean_text(
        chunk
    )

    question_words = [
        word
        for word in question_clean.split()
        if word not in STOP_WORDS
        and len(word) >= 3
    ]

    score = 0

    for word in question_words:

        if word in chunk_clean:

            score += 1

    return score


# =========================================
# FIND RELEVANT CHUNKS
# =========================================

def find_relevant_chunks(
    question,
    chunks
):

    scored_chunks = []

    for chunk in chunks:

        score = score_chunk(
            question,
            chunk
        )

        if score > 0:

            scored_chunks.append(
                (
                    score,
                    chunk
                )
            )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    if scored_chunks:

        return [
            chunk
            for score, chunk
            in scored_chunks[:2]
        ]

    return chunks[:1]


# =========================================
# CUSTOM CSS
# =========================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="text-align:center;">
            📚 Knowledge Base
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Your personal learning resources"
    )

    st.success(
        f"📚 {len(knowledge_files)} knowledge files available"
    )

    st.divider()

    for file in knowledge_files:

        st.markdown(
            f"📄 **{file.name}**"
        )

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown(
        "### 💡 Quick Start"
    )

    st.caption(
        "Try asking:"
    )

    st.markdown(
        """
        • What is Python?  
        • What is Machine Learning?  
        • What is Deep Learning?  
        • What is Data Science?  
        • What is Generative AI?
        """
    )

    st.divider()

    st.caption(
        "🤖 Powered by Gemini"
    )


# =========================================
# MAIN HEADER
# =========================================

st.markdown(
    """
    <div class="main-title">
        🤖 Personal AI Knowledge Base
    </div>

    <div class="subtitle">
        Ask questions and get intelligent answers
        from your personal knowledge.
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "💡 Ask a question about Python, Machine Learning, "
    "Deep Learning, Data Science, or Generative AI."
)


# =========================================
# CHAT HISTORY
# =========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if not st.session_state.messages:

    st.info(
        "👋 Welcome! Ask me anything about your knowledge base."
    )


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        if "source" in message:

            st.caption(
                f"📄 Source: {message['source']}"
            )


# =========================================
# CHAT INPUT
# =========================================

question = st.chat_input(
    "Ask a question about your knowledge base..."
)


if question:

    # =========================================
    # USER MESSAGE
    # =========================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.write(question)


    # =========================================
    # FIND FILE
    # =========================================

    selected_file = find_file(
        question
    )


    if selected_file is None:

        answer = (
            "I couldn't find this topic "
            "in your knowledge base."
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


    # =========================================
    # READ FILE
    # =========================================

    try:

        content = selected_file.read_text(
            encoding="utf-8"
        )

    except Exception as error:

        st.error(
            f"❌ Could not read file: {error}"
        )

        st.stop()


    # =========================================
    # RETRIEVE KNOWLEDGE
    # =========================================

    chunks = create_chunks(
        content
    )

    relevant_chunks = find_relevant_chunks(
        question,
        chunks
    )

    context = "\n\n".join(
        relevant_chunks
    )


    # =========================================
    # CREATE PROMPT
    # =========================================

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the knowledge provided below.

Knowledge:
{context}

User Question:
{question}

Instructions:
- Use the provided knowledge.
- Give a clear and simple answer.
- Give a concise and easy-to-understand response.
- If the knowledge does not contain the answer, say that the information is not available in the knowledge base.
"""


    # =========================================
    # GEMINI
    # =========================================

    try:

        with st.chat_message("assistant"):

            with st.spinner("🤖 Thinking..."):

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                answer = response.text

            st.write(answer)

            st.caption(
                f"📄 Source: {selected_file.name}"
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source": selected_file.name
            }
        )


    except Exception as error:

        st.error(
            f"❌ Gemini Error: {error}"
        )