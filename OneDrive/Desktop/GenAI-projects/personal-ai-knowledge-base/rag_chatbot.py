from pathlib import Path
from difflib import get_close_matches
import re
import os

from dotenv import load_dotenv
from google import genai


# =========================================
# GEMINI SETUP
# =========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Gemini API key not found!")
    print("Please check your .env file.")
    exit()

client = genai.Client(api_key=api_key)


# =========================================
# PERSONAL AI KNOWLEDGE BASE
# =========================================

KNOWLEDGE_FOLDER = Path("knowledge")
CHUNK_SIZE = 50


print("\n" + "=" * 60)
print("🤖 PERSONAL AI KNOWLEDGE BASE - RAG CHATBOT")
print("=" * 60)


# =========================================
# LOAD KNOWLEDGE FILES
# =========================================

knowledge_files = list(
    KNOWLEDGE_FOLDER.glob("*.txt")
)

if not knowledge_files:
    print("\n❌ No knowledge files found!")
    exit()


print("\n📚 Knowledge files:")

for file in knowledge_files:
    print("   •", file.name)

print("\nType 'exit' to stop.")


# =========================================
# TOPIC MAPPING
# =========================================

topic_files = {}

for file in knowledge_files:

    topic = file.stem.lower()

    topic = topic.replace("_notes", "")
    topic = topic.replace("_", " ")

    topic_files[topic] = file


# Aliases

if "ml" in topic_files:
    topic_files["machine learning"] = topic_files["ml"]

if "dl" in topic_files:
    topic_files["deep learning"] = topic_files["dl"]

if "my" in topic_files:

    topic_files["generative ai"] = topic_files["my"]
    topic_files["generative artificial intelligence"] = topic_files["my"]
    topic_files["llm"] = topic_files["my"]
    topic_files["llms"] = topic_files["my"]
    topic_files["large language model"] = topic_files["my"]
    topic_files["large language models"] = topic_files["my"]


# =========================================
# STOP WORDS
# =========================================

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an",
    "tell", "me", "about", "explain",
    "please", "can", "you", "give",
    "some", "information", "on", "do",
    "know", "i", "want", "to", "how",
    "does", "it", "work", "define",
    "describe", "details", "of", "for",
    "and", "in", "my", "your",
    "where", "why", "when", "which",
    "used", "use", "uses", "its",
    "something"
}


# =========================================
# CLEAN TEXT
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

    question_clean = clean_text(question)

    topics = sorted(
        topic_files.keys(),
        key=len,
        reverse=True
    )

    # Exact topic match

    for topic in topics:

        if topic in question_clean:

            return topic_files[topic]


    # Spelling support

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

            return topic_files[matches[0]]


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
            words[i:i + CHUNK_SIZE]
        )

        if chunk.strip():

            chunks.append(chunk)


    return chunks


# =========================================
# SCORE CHUNKS
# =========================================

def score_chunk(question, chunk):

    question_clean = clean_text(question)
    chunk_clean = clean_text(chunk)

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

def find_relevant_chunks(question, chunks):

    scored_chunks = []

    for chunk in chunks:

        score = score_chunk(
            question,
            chunk
        )

        if score > 0:

            scored_chunks.append(
                (score, chunk)
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
# MEMORY
# =========================================

last_file = None


# =========================================
# CHAT LOOP
# =========================================

while True:

    question = input("\nYou: ").strip()


    # EXIT

    if question.lower() == "exit":

        print("\n🤖 Bot: Goodbye! 👋")
        break


    if not question:

        print("🤖 Bot: Please ask a question.")

        continue


    # =====================================
    # FIND KNOWLEDGE FILE
    # =====================================

    selected_file = find_file(question)


    # =====================================
    # FOLLOW-UP QUESTION
    # =====================================

    if selected_file is None:

        follow_up_words = {
            "it",
            "this",
            "that",
            "they",
            "them",
            "where",
            "why",
            "how",
            "when",
            "which"
        }

        question_words = clean_text(
            question
        ).split()


        is_follow_up = any(
            word in follow_up_words
            for word in question_words
        )


        if (
            is_follow_up
            and last_file is not None
        ):

            selected_file = last_file


    # =====================================
    # TOPIC NOT FOUND
    # =====================================

    if selected_file is None:

        print(
            "\n❌ Bot: I couldn't find this topic "
            "in your knowledge base."
        )

        print("\n💡 Available topics:")

        displayed = set()

        for topic, file in topic_files.items():

            if file.name in displayed:
                continue

            print("   •", topic.title())

            displayed.add(file.name)

        continue


    # Remember topic

    last_file = selected_file


    # =====================================
    # READ KNOWLEDGE FILE
    # =====================================

    try:

        content = selected_file.read_text(
            encoding="utf-8"
        )

    except Exception as error:

        print(
            "\n❌ Could not read file:",
            error
        )

        continue


    # =====================================
    # CREATE CHUNKS
    # =====================================

    chunks = create_chunks(content)


    # =====================================
    # RETRIEVE RELEVANT KNOWLEDGE
    # =====================================

    relevant_chunks = find_relevant_chunks(
        question,
        chunks
    )


    # =====================================
    # CREATE CONTEXT
    # =====================================

    context = "\n\n".join(
        relevant_chunks
    )


    # =====================================
    # SEND TO GEMINI
    # =====================================

    prompt = f"""
You are a helpful Personal AI Knowledge Base assistant.

Answer the user's question using ONLY the knowledge
provided below.

Do not invent information.

If the answer is not available in the provided knowledge,
say:

"I couldn't find that information in your knowledge base."

Knowledge:
-------------------------
{context}
-------------------------

User Question:
{question}

Give a clear and simple answer.
"""


    try:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt
        )


        # =================================
        # DISPLAY ANSWER
        # =================================

        print("\n" + "=" * 60)

        print("🤖 Gemini:")

        print(interaction.output_text)


        # =================================
        # SOURCE
        # =================================

        print("\n📚 Knowledge used:")
        print("   📄", selected_file.name)

        print("=" * 60)


    except Exception as error:

        print("\n❌ Gemini Error:")
        print(error)