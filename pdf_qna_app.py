import streamlit as st
import fitz
import io
import asyncio
import httpx
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import re

st.set_page_config(page_title="QnAfy AI Extractor", layout="wide")
st.title("📄 QnAfy Pro – AI-Based Question Answering")

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"

# 🧠 Use AI to extract technical questions from raw text
async def extract_questions_with_ai(text, client):
    prompt = f"""Extract all distinct technical interview questions from the following raw PDF content.
Return only a numbered list of actual questions. Ignore section headers, context text, and answers.

Text:
{text}"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"]
    return re.findall(r'\d+\.\s*(.+)', result)

# 🎯 Build answer prompt
def build_prompt(question, style="Concise"):
    style_map = {
        "Concise": "Answer this question briefly:",
        "Detailed": "Explain this in detail:",
        "Step-by-step": "Solve step-by-step:"
    }
    return f"{style_map.get(style)}\n\n{question}"

# 🧠 Ask AI to answer a single question
async def get_answer_async(question, client, style="Concise", retries=2):
    if len(question) < 10:
        return "[Skipped: Too short or unclear]"
    prompt = build_prompt(question, style)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    for attempt in range(retries + 1):
        try:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.RequestError as e:
            if attempt == retries:
                return f"[Error: {e}]"
            await asyncio.sleep(2 * (attempt + 1))

# 🔁 Get answers in order
async def answer_questions_async(questions, style):
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [get_answer_async(q, client, style=style) for q in questions]
        return await asyncio.gather(*tasks)  # preserves order

# 📤 PDF Upload
uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])
style_choice = st.selectbox("🎯 Answer Style", ["Concise", "Detailed", "Step-by-step"])

# 🧠 AI Question Extraction
if uploaded_file and "ai_questions" not in st.session_state:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    st.info("🧠 Extracting questions using AI...")

    async def run_extraction(text):
        async with httpx.AsyncClient(timeout=90.0) as client:
            return await extract_questions_with_ai(text, client)

    st.session_state.ai_questions = asyncio.run(run_extraction(full_text))
    st.rerun()

# 📝 Show extracted questions
if "ai_questions" in st.session_state:
    questions = st.session_state.ai_questions
    st.subheader("✅ AI-Extracted Questions")
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}:** {q}")

    if st.button("🚀 Generate Answers"):
        answers = asyncio.run(answer_questions_async(questions, style_choice))
        st.session_state.qa = list(zip(questions, answers))
        st.rerun()

# ✅ Show answer previews
if "qa" in st.session_state and st.session_state.qa:
    st.success("✅ Answers generated.")
    st.subheader("📋 Preview Q&A")
    for i, (q, a) in enumerate(st.session_state.qa):
        with st.expander(f"Q{i+1}"):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")

    # 📄 Export as PDF
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    y = height - 50

    for i, (q, a) in enumerate(st.session_state.qa):
        for line in [f"Q{i+1}: {q}", f"A: {a}"]:
            for subline in line.split('\n'):
                pdf.drawString(50, y, subline)
                y -= 15
                if y < 100:
                    pdf.showPage()
                    y = height - 50
        y -= 20

    pdf.save()
    output.seek(0)
    st.download_button("📥 Download Q&A PDF", output, file_name="QnAfy_Final_Answers.pdf", mime="application/pdf")
