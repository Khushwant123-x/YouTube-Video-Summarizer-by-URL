from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

llm = ChatMistralAI(model = "mistral-small-2506")
ytt_api = YouTubeTranscriptApi()
URL = input("Enter the Youtube video URL:")

def get_video_id(url):
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query)["v"][0]
    elif "youtu.be" in parsed.netloc:
        return parsed.path[1:]
    return None

video_id = get_video_id(URL)
transcript = ytt_api.fetch(video_id)
full_text = " ".join(item.text for item in transcript)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(full_text)
# print(len(chunks))
# print(chunks[0][:1000])

prompt = ChatPromptTemplate.from_template("""
You are an expert YouTube video summarizer.

Summarize the following transcript.

Rules:
- Keep important points
- Use bullet points
- Remove filler words
- Maximum 200 words

Transcript:
{text}
""")

partial_summaries = []

for i, chunk in enumerate(chunks):

    print(f"Summarizing chunk {i+1}/{len(chunks)}")

    formatted_prompt = prompt.invoke({
        "text": chunk
    })

    result = llm.invoke(
        formatted_prompt
    )

    partial_summaries.append(
        result.content
    )

combined_summary = "\n".join(
    partial_summaries
)

final_prompt = ChatPromptTemplate.from_template("""
Combine these summaries into one final concise summary.

Summaries:
{text}
""")

formatted_final = final_prompt.invoke({
    "text": combined_summary
})

final_summary = llm.invoke(
    formatted_final
)
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(final_summary.content)
