import streamlit as st
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import re
import asyncio
import httpx
import time

st.set_page_config(page_title="QnAfy: PDF Q&A with AI", layout="centered")
st.title("📄 QnAfy – AI-powered PDF Question Answering")

# --- CONFIG ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"  # Free and fast

# --- UI: Upload + Reset ---
uploaded_file = st.file_uploader("📤 Upload a PDF containing questions", type=["pdf"])
if st.button("🔁 Reset / Upload New PDF"):
    st.session_state.clear()
    st.rerun()


# --- Clean Text & Extract Questions ---
def fix_text_and_extract_questions(text):
    # Fix camelCase and broken joint words
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=\w)([&])(?=\w)", " & ", text)

    # Normalize and merge lines
    lines = text.replace('\r', '').split('\n')
    merged = ' '.join([line.strip() for line in lines if line.strip()])

    # Extract questions ending with '?'
    raw_questions = [q.strip() + '?' for q in merged.split('?') if len(q.strip()) > 5]
    return raw_questions


# --- Async OpenRouter QnA ---
async def get_answer_async(question, client):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": f"Answer this question:\n\n{question}"}]
    }

    try:
        response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {e}]"

async def answer_questions_async(questions):
    results = []
    progress = st.progress(0, "⏳ Answering...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [get_answer_async(q, client) for q in questions]
        total = len(tasks)
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            a = await coro
            results.append(a)
            progress.progress((i + 1) / total, f"Answered {i+1}/{total}")
    return results


# --- Main Logic ---
if uploaded_file:
    if "last_filename" not in st.session_state or st.session_state.last_filename != uploaded_file.name:
        st.session_state.last_filename = uploaded_file.name
        st.session_state.qna_done = False
        st.session_state.all_qna = []

        st.info("📖 Reading your PDF...")
        pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        all_questions = []

        for page_num, page in enumerate(pdf_reader, start=1):
            raw_text = page.get_text()
            questions = fix_text_and_extract_questions(raw_text)
            if questions:
                st.markdown(f"- Page {page_num}: found `{len(questions)}` question(s)")
                all_questions.extend(questions)

        if not all_questions:
            st.warning("❌ No valid questions found in this PDF.")
        else:
            est_time = len(all_questions) * 3
            st.caption(f"🕒 Estimated time: ~{est_time} seconds for {len(all_questions)} question(s)")

            answers = asyncio.run(answer_questions_async(all_questions))
            st.session_state.all_qna = list(zip(all_questions, answers))
            st.session_state.qna_done = True
            st.success("✅ All questions answered! You can now download the final PDF.")

    all_qna = st.session_state.all_qna

    # Generate and serve PDF
    if st.session_state.qna_done:
        output = io.BytesIO()
        pdf = canvas.Canvas(output, pagesize=letter)
        width, height = letter
        y = height - 50

        for q, a in all_qna:
            for line in [f"Q: {q}", f"A: {a}"]:
                for subline in line.split('\n'):
                    pdf.drawString(50, y, subline)
                    y -= 15
                    if y < 100:
                        pdf.showPage()
                        y = height - 50
            y -= 20

        pdf.save()
        output.seek(0)

        st.download_button("📥 Download Answered PDF", output, file_name="Answered_PDF.pdf", mime="application/pdf")
