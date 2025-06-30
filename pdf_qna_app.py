import streamlit as st
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import requests
import re

st.set_page_config(page_title="PDF Q&A Generator", layout="centered")
st.title("📄 PDF Q&A Generator with AI (via OpenRouter)")

# --- CONFIG ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"  # Use a free model for now

uploaded_file = st.file_uploader("📤 Upload a PDF containing questions", type=["pdf"])

if st.button("🔁 Reset / Upload New PDF"):
    st.session_state.clear()
    st.rerun()

def get_answer(question):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": f"Answer the following question briefly:\n\n{question}"}
        ]
    }
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {e}]"

# --- Text Cleaning ---
def fix_text_and_extract_questions(text):
    # Fix mid-word camel case (no space between words)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=\w)([&])(?=\w)", " & ", text)  # Fix things like C&DataStructures

    # Normalize line endings
    lines = text.replace('\r', '').split('\n')
    combined = ' '.join([line.strip() for line in lines if line.strip()])

    # Extract questions by splitting on '?'
    raw_questions = [q.strip() + '?' for q in combined.split('?') if len(q.strip()) > 5 and '?' not in q.strip()]
    
    return raw_questions

if uploaded_file:
    if "last_filename" not in st.session_state or st.session_state.last_filename != uploaded_file.name:
        st.session_state.last_filename = uploaded_file.name
        st.session_state.qna_done = False
        st.session_state.all_qna = []

        st.info("⏳ Reading and analyzing PDF...")

        pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        all_qna = []

        for page_num, page in enumerate(pdf_reader, start=1):
            text = page.get_text()
            questions = fix_text_and_extract_questions(text)

            if not questions:
                continue

            with st.spinner(f"🔍 Page {page_num}: answering {len(questions)} question(s)..."):
                for q in questions:
                    a = get_answer(q)
                    all_qna.append((q, a))

        st.session_state.all_qna = all_qna
        st.session_state.qna_done = True
        st.success("✅ All questions answered. Ready to download!")

    all_qna = st.session_state.all_qna

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

        st.download_button("📥 Download Enhanced PDF", output, file_name="Answered_PDF.pdf", mime="application/pdf")
