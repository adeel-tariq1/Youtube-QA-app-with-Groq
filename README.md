YouTube Q&A and Summarization App with Groq 🎥🤖
A powerful Streamlit application that transforms YouTube videos into interactive learning experiences. Extract audio from any YouTube video and get instant AI-powered answers or summaries using Groq's lightning-fast LLMs.

✨ Features
🎯 Core Functionality
YouTube Audio Extraction: Download and process audio from any YouTube video

AI-Powered Q&A: Ask questions about video content and get precise answers

Smart Summarization: Generate concise summaries of video content

Fast Transcription: Uses Groq's Whisper API for quick audio-to-text conversion

🚀 Technical Highlights
Lightning Fast Responses: Powered by Groq's Llama 3.1 8B instant model

Semantic Search: ChromaDB vector store with SentenceTransformer embeddings

Local Processing: No data sent to external servers for embeddings

User-Friendly Interface: Clean Streamlit UI with real-time progress updates

🛠️ Installation
Prerequisites
Python 3.8+

Groq API Key (Get it here)

FFmpeg (for audio processing)

Quick Start

Clone the repository

bash
git clone https://github.com/yourusername/youtube-ai-assistant.git
cd youtube-ai-assistant
Install dependencies

bash
pip install -r requirements.txt

Run the application

bash
streamlit run app.py
📖 Usage
Step 1: Configure API Key
Enter your Groq API key in the sidebar

Select between "Q&A" or "Summarize" mode

Step 2: Provide YouTube URL
Paste any YouTube video URL in the main input field

Step 3: Choose Your Action

 Q&A Mode
Click "Extract Audio and Transcribe"

Wait for processing to complete

Ask questions about the video content in natural language

Get answers with source references from the transcription

📝 Summarize Mode
Click "Summarize Video"

Get an AI-generated summary of the main points

View full transcription for reference
<img width="6836" height="310" alt="deepseek_mermaid_20250928_6a7018" src="https://github.com/user-attachments/assets/f3d3c585-64c7-41ab-b828-b224d2b4a506" />

Model Settings
Transcription: whisper-large-v3

Q&A LLM: llama-3.1-8b-instant

Embeddings: all-MiniLM-L6-v2

Vector Store: ChromaDB with SQLite
🤝 Contributing
We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

