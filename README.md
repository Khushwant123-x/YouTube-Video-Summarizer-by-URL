# 🎥 VideoBrief AI — YouTube Video Summarizer

An AI-powered web application that turns any YouTube video into clear, structured summaries and transcripts in seconds using **Groq** ultra-fast LLM inference and **LangChain**.

---

## ✨ Features
- ⚡ **Ultra-Fast Summarization**: Powered by Groq's LPUs and cutting-edge language models.
- 🎯 **Direct YouTube Integration**: Fetches transcripts automatically from standard, shortened (`youtu.be`), and YouTube Shorts URLs.
- 📺 **Interactive Video Preview**: Clean embedded video player.
- 📜 **Full Transcript Access**: View and copy the complete raw transcript.
- 💾 **Export Options**: 1-click download of summaries as Markdown (`.md`) or Plain Text (`.txt`).
- 💎 **Modern Dark UI**: Glassmorphic card styling, vibrant accents, and clean typography.

---

## 🛠️ Tech Stack
- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **LLM Orchestration**: [LangChain](https://www.langchain.com/) & [langchain-groq](https://github.com/langchain-ai/langchain-groq)
- **Transcription**: [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- **LLM Provider**: [Groq](https://groq.com/)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Khushwant123-x/YouTube-Video-Summarizer-by-URL.git
cd YouTube-Video-Summarizer-by-URL
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(You can get a free API key at [console.groq.com](https://console.groq.com/))*

### 5. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
