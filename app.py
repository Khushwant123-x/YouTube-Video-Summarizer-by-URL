import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()

# LLM
llm = ChatMistralAI(
    model="mistral-small-2506"
)

ytt_api = YouTubeTranscriptApi()


# -----------------------------
# Extract Video ID
# -----------------------------
def get_video_id(url):
    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query)["v"][0]

    elif "youtu.be" in parsed.netloc:
        return parsed.path[1:]

    return None


# -----------------------------
# Summarizer Function
# -----------------------------
def summarize_video(url):

    video_id = get_video_id(url)

    transcript = ytt_api.fetch(video_id)

    full_text = " ".join(
        item.text for item in transcript
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(
        full_text
    )

    prompt = ChatPromptTemplate.from_template("""
You are an expert YouTube summarizer.

Summarize transcript.

Rules:
- Bullet points
- Remove filler
- Maximum 200 words

Transcript:
{text}
""")

    partial = []

    for chunk in chunks:

        formatted = prompt.invoke({
            "text": chunk
        })

        result = llm.invoke(
            formatted
        )

        partial.append(
            result.content
        )

    combined = "\n".join(partial)

    final_prompt = ChatPromptTemplate.from_template("""
Create one final summary.

{text}
""")

    formatted_final = final_prompt.invoke({
        "text": combined
    })

    final = llm.invoke(
        formatted_final
    )

    return final.content


# -----------------------------
# UI
# -----------------------------

st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube Video Summarizer")

st.write(
    "Paste YouTube URL and get AI summary"
)

url = st.text_input(
    "Enter YouTube URL"
)

if st.button("Generate Summary"):

    if not url:
        st.warning(
            "Enter URL first"
        )

    else:
        try:

            with st.spinner(
                "Generating summary..."
            ):

                summary = summarize_video(
                    url
                )

            st.success(
                "Summary Ready"
            )

            st.subheader(
                "📌 Final Summary"
            )

            st.write(
                summary
            )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )