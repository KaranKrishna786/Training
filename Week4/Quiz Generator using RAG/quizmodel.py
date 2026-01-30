# app.py
# Streamlit UI for AI-Powered Quiz Generator (LangChain + TinyLlama/HF)
# Author: Karan-ready
# Run: streamlit run app.py

import streamlit as st
import torch

from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="AI Quiz Generator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI-Powered Quiz Generator")
st.caption("Generate quizzes from any subject using a local open-source LLM (TinyLlama + LangChain).")

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    model_id = st.text_input(
        "Model ID (Hugging Face Hub)",
        value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="You can replace with any compatible chat/causal LM on Hugging Face."
    )

    quiz_style = st.selectbox(
        "Quiz Style",
        options=["Questions only", "MCQs (with answers)", "Short answers (with key)"],
        index=1
    )

    difficulty = st.selectbox(
        "Difficulty",
        options=["Beginner", "Intermediate", "Advanced"],
        index=1
    )

    n_questions = st.slider("Number of questions", min_value=5, max_value=20, value=10, step=1)

    st.markdown("---")
    st.subheader("Generation Controls")

    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.3, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.05, 0.01)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 400, 16)

    use_gpu = st.checkbox("Use GPU if available", value=True)
    seed = st.number_input("Random seed (optional)", value=42, step=1, help="For reproducibility. Set None to disable.")

# ---------------------------
# Helper: Build Prompt by Style
# ---------------------------
def build_prompt_template(style: str) -> PromptTemplate:
    """
    Use TinyLlama chat format. We’ll add a short system message to guide outputs.
    """
    if style == "Questions only":
        template = (
            "<|system|>\n"
            "You are a concise, expert quiz author. Generate only the questions, "
            "numbered 1..N. No answers.\n</s>\n"
            "<|user|>\n"
            "Topic: {course}\n"
            "Difficulty: {difficulty}\n"
            "Generate exactly {n} high-quality quiz questions as a numbered list. "
            "Avoid duplication and ensure coverage of key concepts.\n"
            "<|assistant|>\n"
        )
    elif style == "MCQs (with answers)":
        template = (
            "<|system|>\n"
            "You are a precise exam designer. Write MCQs with four options (A–D). "
            "After listing all questions, provide an answer key.\n</s>\n"
            "<|user|>\n"
            "Topic: {course}\n"
            "Difficulty: {difficulty}\n"
            "Generate exactly {n} multiple-choice questions. For each question, include options A–D. "
            "After all questions, add a section 'Answer Key' listing the correct option for each number.\n"
            "Format strictly:\n"
            "1) <question>\n\n\n\n"
            "   A) ...\n"
            "   B) ...\n"
            "   C) ...\n"
            "   D) ...\n"
            "...\n"
            "Answer Key:\n"
            "1) <A/B/C/D>\n"
            "2) <A/B/C/D>\n"
            "...\n"
            "<|assistant|>\n"
        )
    else:  # "Short answers (with key)"
        template = (
            "<|system|>\n"
            "You are a concise quiz generator. Provide short-answer questions and then an answer key.\n</s>\n"
            "<|user|>\n"
            "Topic: {course}\n"
            "Difficulty: {difficulty}\n"
            "Generate exactly {n} short-answer questions as a numbered list. "
            "After the questions, add a section 'Answer Key' with brief, accurate answers.\n"
            "<|assistant|>\n"
        )

    return PromptTemplate.from_template(template)

# ---------------------------
# Cached Model Loader
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_llm_cached(
    model_id: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    use_gpu: bool
) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Decide device
    device = 0 if (use_gpu and torch.cuda.is_available()) else -1

    gen_pipe = pipeline(
        task="text-generation",
        model=model_id,
        tokenizer=tokenizer,
        device=device,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        return_full_text=False,   # Only generated continuation
    )

    return HuggingFacePipeline(pipeline=gen_pipe)

# ---------------------------
# Main UI
# ---------------------------
subject = st.text_input("Enter the subject/topic", placeholder="e.g., Python basics, Linear Regression, Workplace Safety")

col1, col2 = st.columns(2)
with col1:
    generate_btn = st.button("🚀 Generate Quiz", type="primary")
with col2:
    clear_btn = st.button("🧹 Clear")

if clear_btn:
    st.session_state.pop("quiz_text", None)
    st.rerun()

if generate_btn:
    if not subject.strip():
        st.warning("Please enter a subject/topic.")
        st.stop()

    # Optional: seed
    if seed is not None:
        try:
            import random
            random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass

    with st.spinner("Loading model & generating quiz... (first time may take a minute)"):
        llm = load_llm_cached(
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            use_gpu=use_gpu
        )

        prompt = build_prompt_template(quiz_style)
        chain = prompt | llm | StrOutputParser()

        quiz_text = chain.invoke({
            "course": subject.strip(),
            "difficulty": difficulty,
            "n": n_questions
        })

        st.session_state["quiz_text"] = quiz_text

# Display output (if available)
if "quiz_text" in st.session_state:
    st.markdown("### 📋 Your Quiz")
    st.markdown(st.session_state["quiz_text"])

    # Download button
    st.download_button(
        label="💾 Download as TXT",
        data=st.session_state["quiz_text"],
        file_name=f"quiz_{subject.replace(' ', '_')}.txt",
        mime="text/plain"
    )

# Footer note
st.markdown("---")
st.caption(
    "Tip: For faster inference, use a GPU-enabled Python environment. "
    "TinyLlama (1.1B) should work on CPU but may be slower."
)