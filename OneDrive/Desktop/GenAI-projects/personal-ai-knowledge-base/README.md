# 🤖 Personal AI Knowledge Base

A GenAI-powered personal knowledge assistant built using Python, Streamlit, and Google Gemini.

This application allows users to ask questions and receive simple AI-generated answers based on information stored in their personal knowledge base.

## 🚀 Live Demo

https://personal-ai-knowledge-base-cttfvpf8wt6gbm7bajm3ns.streamlit.app/

## ✨ Features

- 🤖 AI-powered question answering
- 📚 Personal knowledge base
- 🔎 Relevant knowledge retrieval
- 💬 Interactive chat interface
- 📄 Source file display
- 📑 TXT file support
- 📕 PDF file support
- 🐍 Python knowledge
- 🤖 Machine Learning knowledge
- 🧠 Deep Learning knowledge
- 📊 Data Science knowledge
- ✨ Generative AI knowledge
- 🧹 Clear chat option
- ☁️ Deployed using Streamlit

## 🛠️ Technologies Used

- Python
- Streamlit
- Google Gemini API
- PyPDF
- python-dotenv

## 🧠 How It Works

1. User enters a question.
2. The application identifies the relevant knowledge file.
3. The relevant content is retrieved from the knowledge base.
4. The retrieved information is provided to Google Gemini.
5. Gemini generates a clear and simple answer.
6. The application displays the answer along with the source file.

## 📂 Project Structure

```text
Personal-AI-Knowledge-Base/
│
├── app.py
├── requirements.txt
├── .gitignore
├── knowledge/
│   ├── python_notes.txt
│   ├── ml_notes.txt
│   ├── deep_learning_notes.txt
│   ├── data_science_notes.txt
│   ├── college_notes.txt
│   ├── my_notes.txt
│   └── Supervised-Machine-Learning.txt
│
└── README.md
▶️ Run Locally

Clone the repository:

git clone https://github.com/abhinayapodakatla/Personal-AI-Knowledge-Base.git

Go to the project folder:

cd Personal-AI-Knowledge-Base

Install the required packages:

pip install -r requirements.txt

Create a .env file and add your Gemini API key:

GEMINI_API_KEY=your_api_key_here

Run the application:

streamlit run app.py
🎯 Project Goal

The goal of this project is to build a simple personal AI assistant that can understand questions and provide answers using information from a customized knowledge base.

🔮 Future Improvements
Add more advanced RAG techniques
Support more document formats
Improve semantic search using embeddings
Add conversation memory
Add user authentication
Improve answer accuracy
Add a larger personal knowledge base
👨‍💻 Author

Abhinaya

B.Tech CSE (AI/ML)
