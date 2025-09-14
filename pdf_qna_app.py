import streamlit as st
import fitz  # PyMuPDF
import io
import asyncio
import httpx
import json
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib.colors import navy, black

# --- Page Configuration ---
st.set_page_config(page_title="QnAfy AI Extractor", layout="wide", initial_sidebar_state="collapsed")
st.title("📄 QnAfy Pro – AI-Based Question & Answer Extractor")

# --- Constants & Secrets ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
AVAILABLE_MODELS = {
    "Mistral 7B Instruct": "mistralai/mistral-7b-instruct",
    "Llama 3 8B Instruct": "meta-llama/llama-3-8b-instruct",
    "Mythomist 7B": "gryphe/mythomist-7b"
}

# --- Core AI Functions ---

async def make_ai_request(client, model, messages, retries=2):
    """A robust, reusable function to make API calls to OpenRouter."""
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    # Request JSON format for reliable parsing
    data = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
    
    for attempt in range(retries + 1):
        try:
            response = await client.post(API_URL, json=data, headers=headers)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            # The API sometimes returns the JSON string within a markdown block, so we extract it.
            json_match = re.search(r'```json\n({.*})\n```', content, re.S)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(content) # Directly return the parsed JSON
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as e:
            if attempt == retries:
                st.error(f"API Error after multiple retries: {e}.")
                return None
            await asyncio.sleep(2 * (attempt + 1)) # Exponential backoff

async def extract_questions_with_ai(text, client, model):
    """Uses AI to extract questions from text, expecting a JSON response."""
    prompt = f"""
    Analyze the following text from a PDF document. Your task is to identify and extract all distinct technical interview questions.

    Please adhere to these rules:
    1. Ignore all non-question text, such as section titles, introductory paragraphs, and answers.
    2. Extract only the questions themselves.
    3. Return the output as a single JSON object with a single key "questions", which contains a list of the extracted question strings.

    Example output format:
    {{
      "questions": [
        "What is the difference between a list and a tuple in Python?",
        "Explain the concept of polymorphism in object-oriented programming."
      ]
    }}

    Text to analyze (first 8000 characters):
    ---
    {text[:8000]}
    """
    messages = [{"role": "user", "content": prompt}]
    result = await make_ai_request(client, model, messages)
    
    if result and "questions" in result and isinstance(result["questions"], list):
        return result["questions"]
    st.warning("AI failed to extract questions in the expected format. No questions were found.")
    return []

async def get_single_answer(question, client, model, style):
    """Gets an AI-generated answer for a single question, expecting a JSON response."""
    if not question or len(question) < 10:
        return "[Skipped: Question was too short or empty]"
        
    style_instructions = {
        "Concise": "Provide a clear and concise answer, typically in 2-3 sentences.",
        "Detailed": "Explain the concept in detail, covering key aspects and providing context. Use paragraphs for structure.",
        "Step-by-step": "Provide a step-by-step explanation or solution. Use a numbered list if appropriate."
    }
    
    prompt = f"""
    You are an expert technical interviewer. Your task is to answer the following question in a helpful and accurate manner.

    Question:
    "{question}"

    Instructions:
    - Answer according to the requested style: **{style}**.
    - {style_instructions.get(style, "")}
    - The output must be a single JSON object with one key, "answer", containing the string response.
    
    Example output format:
    {{
      "answer": "The answer to the question goes here..."
    }}
    """
    messages = [{"role": "user", "content": prompt}]
    result = await make_ai_request(client, model, messages)
    
    if result and "answer" in result:
        return result["answer"]
    return "[Error: Failed to get a valid answer from the AI]"

async def run_answer_generation():
    """FIXED: Generates all answers concurrently inside the client context."""
    questions = st.session_state.ai_questions
    model = AVAILABLE_MODELS[st.session_state.model_choice_key]
    style = st.session_state.style_choice
    
    # Create the client and perform all network operations INSIDE this block
    async with httpx.AsyncClient(timeout=90.0) as client:
        tasks = [get_single_answer(q, client, model, style) for q in questions]
        # Run all tasks concurrently and wait for them all to complete
        answers = await asyncio.gather(*tasks)
    
    st.session_state.qa = list(zip(questions, answers))

# --- PDF Generation with Text Wrapping ---

def create_qa_pdf(qa_pairs):
    """Generates a PDF from a list of question-answer pairs with proper text wrapping."""
    buffer = io.BytesIO()
    doc = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 72
    
    styles = getSampleStyleSheet()
    style_q = styles['h3']
    style_q.textColor = navy
    style_a = styles['BodyText']
    style_a.textColor = black

    y_pos = height - margin
    
    for i, (q, a) in enumerate(qa_pairs):
        question_text = f"Q{i+1}: {q}"
        answer_text = f"A: {a}"

        p_q = Paragraph(question_text, style_q)
        p_a = Paragraph(answer_text, style_a)
        
        q_height = p_q.wrapOn(doc, width - 2 * margin, height)[1]
        a_height = p_a.wrapOn(doc, width - 2 * margin, height)[1]
        total_height = q_height + a_height + 25

        if y_pos - total_height < margin:
            doc.showPage()
            y_pos = height - margin

        p_q.drawOn(doc, margin, y_pos - q_height)
        y_pos -= (q_height + 10)
        
        p_a.drawOn(doc, margin, y_pos - a_height)
        y_pos -= (a_height + 15)

    doc.save()
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])
    model_choice_key = st.selectbox("🤖 Choose AI Model", options=list(AVAILABLE_MODELS.keys()))
    style_choice = st.selectbox("🎯 Answer Style", ["Concise", "Detailed", "Step-by-step"])
    
    if st.button("🔄 Start Over"):
        keys_to_clear = ["ai_questions", "qa", "answers_in_progress", "model_choice_key", "style_choice"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Main Page Logic ---

# Step 1: File Upload and Question Extraction
if uploaded_file and "ai_questions" not in st.session_state:
    with st.spinner("Analyzing PDF and extracting questions... 🧠"):
        pdf_bytes = uploaded_file.read()
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = "\n".join([page.get_text() for page in doc])
            
            async def run_extraction():
                model = AVAILABLE_MODELS[model_choice_key]
                async with httpx.AsyncClient(timeout=90.0) as client:
                    return await extract_questions_with_ai(full_text, client, model)

            st.session_state.ai_questions = asyncio.run(run_extraction())
            st.rerun()
        except Exception as e:
            st.error(f"Error processing PDF. It might be corrupted or unreadable. Details: {e}")

# Step 2: Display Extracted Questions and Offer to Generate Answers
if "ai_questions" in st.session_state and not st.session_state.get("answers_in_progress") and not st.session_state.get("qa"):
    questions = st.session_state.ai_questions
    if not questions:
        st.warning("No technical questions could be extracted from the document. Please try another PDF or a different model.")
    else:
        st.subheader(f"✅ Extracted {len(questions)} Questions")
        with st.expander("Click to view/edit extracted questions"):
            for i, q in enumerate(questions):
                st.markdown(f"**{i+1}.** {q}")

        if st.button("🚀 Generate Answers", type="primary"):
            st.session_state.answers_in_progress = True
            # Store user's choices in session state to be accessible by the async function
            st.session_state.model_choice_key = model_choice_key 
            st.session_state.style_choice = style_choice
            st.rerun()

# Step 3: Run Answer Generation Process
if st.session_state.get("answers_in_progress"):
    with st.spinner("AI is generating answers... This may take a moment. ⏳"):
        asyncio.run(run_answer_generation())
        st.session_state.answers_in_progress = False
        st.rerun()

# Step 4: Display Q&A and Download Button
if "qa" in st.session_state and st.session_state.qa:
    st.success("✅ Answers generated successfully!")
    st.subheader("📋 Preview Q&A")
    
    for i, (q, a) in enumerate(st.session_state.qa):
        with st.expander(f"**Q{i+1}:** {q[:100]}..."):
            st.markdown(f"**Question:**\n{q}")
            st.markdown("---")
            st.markdown(f"**Answer:**\n{a}")
    
    pdf_buffer = create_qa_pdf(st.session_state.qa)
    st.download_button(
        label="📥 Download Q&A PDF",
        data=pdf_buffer,
        file_name="QnAfy_Generated_Answers.pdf",
        mime="application/pdf",
        type="primary"
            )
