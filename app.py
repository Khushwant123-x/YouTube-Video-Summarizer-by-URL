import streamlit as st
import os
import re
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------------------------
# Configuration & Environment
# -----------------------------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "")

st.set_page_config(
    page_title="VideoBrief AI — YouTube Video Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Modern High-End Styling (Glassmorphism & Neon Glow)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ambient background gradient */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(121, 40, 202, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 85% 20%, rgba(255, 0, 128, 0.10) 0%, transparent 40%),
                    radial-gradient(circle at 50% 80%, rgba(0, 194, 255, 0.08) 0%, transparent 50%),
                    #0b0f17;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(22, 30, 46, 0.65);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: border-color 0.3s ease, transform 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }

    /* Gradient Hero Text */
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 40%, #ff4b4b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(255, 0, 128, 0.12);
        color: #ff5599;
        border: 1px solid rgba(255, 0, 128, 0.3);
        margin-bottom: 14px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 650px;
        margin-bottom: 24px;
        line-height: 1.5;
    }



    /* Primary Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ef4444 0%, #ec4899 50%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 20px -2px rgba(239, 68, 68, 0.45) !important;
        transition: all 0.25s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px 0 rgba(236, 72, 153, 0.6) !important;
    }

    /* Input text field enhancement */
    div[data-testid="stTextInput"] input {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(15, 23, 42, 0.8) !important;
        color: #f8fafc !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #ec4899 !important;
        box-shadow: 0 0 0 2px rgba(236, 72, 153, 0.3) !important;
    }

    /* Tab enhancements */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: #94a3b8 !important;
        padding: 8px 16px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f8fafc !important;
        border-bottom-color: #ef4444 !important;
    }

    /* Code block & transcript container */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Video Extraction & Processing Utilities
# -----------------------------------------------------------------------------
def extract_video_id(url: str):
    """Robustly extract YouTube video ID from various URL patterns."""
    if not url:
        return None
    url = url.strip()
    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]
        # Check shorts, embed, live, v
        paths = [p for p in parsed.path.split("/") if p]
        if len(paths) >= 2 and paths[0] in ["shorts", "embed", "v", "live"]:
            return paths[1]
    elif "youtu.be" in parsed.netloc:
        paths = [p for p in parsed.path.split("/") if p]
        if paths:
            return paths[0].split("?")[0]

    # Raw 11-char ID check fallback
    match = re.search(r"([a-zA-Z0-9_-]{11})", url)
    if match and len(url) <= 15:
        return match.group(1)

    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_transcript(video_id: str):
    """Fetches video transcript using YouTubeTranscriptApi."""
    ytt = YouTubeTranscriptApi()
    transcript_snippets = ytt.fetch(video_id)
    full_text = " ".join(snippet.text for snippet in transcript_snippets)
    return full_text, transcript_snippets


def create_summary(full_text: str, model_name: str, summary_style: str, length_words: int, target_lang: str):
    """Summarizes transcript using chunking and Groq LLM."""
    llm = ChatGroq(
        model=model_name,
        temperature=0.3
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250
    )
    chunks = text_splitter.split_text(full_text)

    # Style instructions
    style_prompts = {
        "Executive Brief": "High-level executive briefing with core thesis, 3 strategic insights, and key takeaways.",
        "Detailed Breakdown": "In-depth structured breakdown with section headings, detailed bullet points, and nuance.",
        "Quick Actionable Bullets": "Ultra-concise, high-impact bullet points focusing strictly on actionable knowledge.",
        "Study Notes & Glossary": "Structured learning notes: Core Concepts, Key Insights, and important terms or definitions."
    }
    style_instruction = style_prompts.get(summary_style, "Comprehensive bullet-point summary.")

    chunk_prompt = ChatPromptTemplate.from_template("""
You are an expert content distiller. Summarize the following portion of a video transcript accurately.
Focus on core ideas, facts, and eliminate conversational filler.

Transcript Section:
{text}

Distilled Notes:
""")

    partial_summaries = []
    progress_bar = st.progress(0, text="Analyzing transcript sections...")

    for idx, chunk in enumerate(chunks):
        formatted = chunk_prompt.invoke({"text": chunk})
        res = llm.invoke(formatted)
        partial_summaries.append(res.content)
        progress_val = int(((idx + 1) / len(chunks)) * 75)
        progress_bar.progress(progress_val, text=f"Processing section {idx + 1} of {len(chunks)}...")

    combined = "\n\n".join(partial_summaries)

    final_prompt = ChatPromptTemplate.from_template("""
You are a world-class executive synthesizer. Create a final, polished summary based on the section notes below.

Configuration:
- Output Style: {style_instruction}
- Target Word Count: approximately {length_words} words
- Language: Respond in {target_lang}

Format Guidelines:
- Use clean Markdown with headers (`##`, `###`), bold keywords, and bullet points.
- Include a quick "⚡ TL;DR" at the very beginning.
- End with a "🎯 Key Takeaway" quote or takeaway.

Section Notes:
{notes}
""")

    progress_bar.progress(85, text="Synthesizing final executive summary...")
    formatted_final = final_prompt.invoke({
        "style_instruction": style_instruction,
        "length_words": length_words,
        "target_lang": target_lang,
        "notes": combined
    })

    final_res = llm.invoke(formatted_final)
    progress_bar.progress(100, text="Summary complete!")
    progress_bar.empty()

    return final_res.content, len(chunks)


# -----------------------------------------------------------------------------
# Sidebar: Controls & Customization
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #ef4444, #8b5cf6); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                ⚡
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700;">VideoBrief AI</h3>
                <p style="margin: 0; font-size: 0.72rem; color: #94a3b8;">Ultra-Fast Video Intelligence</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Default summary configuration
    selected_model = "openai/gpt-oss-120b"
    summary_style = "Executive Brief"
    target_lang = "English"
    length_words = 300
    st.markdown("#### 💡 Quick Test Videos")
    col_demo1, col_demo2 = st.columns(2)
    demo_url = None
    with col_demo1:
        if st.button("Neural Nets", use_container_width=True):
            demo_url = "https://www.youtube.com/watch?v=aircAruvnKk"
    with col_demo2:
        if st.button("Quantum AI", use_container_width=True):
            demo_url = "https://www.youtube.com/watch?v=JhHMJCUmq28"




# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown("""
    <div style="margin-top: 10px;">
        <span class="hero-badge">✦ Instant Video Intelligence</span>
        <h1 class="hero-title">Transform Any YouTube Video into Instant Knowledge</h1>
        <p class="hero-subtitle">
            Extract transcripts, bypass filler content, and generate publication-grade summaries in seconds powered by Groq ultra-fast AI.
        </p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# URL Input & Controls
# -----------------------------------------------------------------------------
if "url_input" not in st.session_state:
    st.session_state["url_input"] = ""

if demo_url:
    st.session_state["url_input"] = demo_url

url = st.text_input(
    "Enter YouTube Video Link",
    value=st.session_state["url_input"],
    placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
    label_visibility="collapsed"
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    generate_clicked = st.button("⚡ Generate Summary", use_container_width=True)
with col_info:
    st.markdown("""
        <div style="padding-top: 8px; font-size: 0.85rem; color: #94a3b8;">
            Paste any link with subtitles or captions to generate.
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Processing & Presentation
# -----------------------------------------------------------------------------
video_id = extract_video_id(url) if url else None

if url and not video_id:
    st.warning("⚠️ Please provide a valid YouTube URL format (e.g. youtube.com/watch?v=... or youtu.be/...)")

# Video Preview Card
if video_id:
    with st.expander("📺 Video Preview", expanded=False):
        st.video(f"https://www.youtube.com/watch?v={video_id}")

if generate_clicked:
    if not url:
        st.warning("Please paste a YouTube URL first!")
    elif not video_id:
        st.error("Could not find a valid YouTube Video ID from the link provided.")
    elif not groq_api_key:
        st.error("🔑 Missing `GROQ_API_KEY`. Please check your `.env` file configuration.")
    else:
        try:
            with st.spinner("📥 Fetching video transcript from YouTube..."):
                transcript_text, raw_snippets = fetch_transcript(video_id)

            if not transcript_text.strip():
                st.error("No transcript or speech text could be extracted from this video.")
            else:
                summary, num_chunks = create_summary(
                    transcript_text,
                    selected_model,
                    summary_style,
                    length_words,
                    target_lang
                )

                st.session_state["last_summary"] = summary
                st.session_state["last_transcript"] = transcript_text
                st.session_state["last_video_id"] = video_id
                st.session_state["num_chunks"] = num_chunks

        except Exception as e:
            error_str = str(e)
            if "TranscriptsDisabled" in error_str or "NoTranscriptFound" in error_str:
                st.error("❌ Transcripts are disabled or unavailable for this video. Please try a video with captions enabled.")
            else:
                st.error(f"❌ Error generating summary: {error_str}")

# Display Summary Results
if "last_summary" in st.session_state and st.session_state["last_video_id"] == video_id:
    summary_text = st.session_state["last_summary"]
    transcript_text = st.session_state["last_transcript"]

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Content Tabs
    tab_summary, tab_transcript, tab_export = st.tabs([
        "📑 Summary",
        "📜 Transcript",
        "💾 Export"
    ])

    with tab_summary:
        st.markdown(f"""
            <div class="glass-card">
                {summary_text}
            </div>
        """, unsafe_allow_html=True)

    with tab_transcript:
        st.markdown("**Full Transcript:**")
        st.text_area(
            "Raw Transcript",
            value=transcript_text,
            height=360,
            label_visibility="collapsed"
        )

    with tab_export:
        st.markdown("### 📥 Save Your Summary")
        export_md = f"# Summary\n\n{summary_text}"

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📄 Download Markdown (.md)",
                data=export_md,
                file_name="summary.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="📝 Download Plain Text (.txt)",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )