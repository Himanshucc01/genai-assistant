import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import random
import cohere
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if not text.strip():
        return "⚠️ No content found in the file to summarize."
    if len(text.split()) < 30:
        return "⚠️ Text too short to summarize."
    if len(text) > 10000:  # Increased limit
        text = text[:10000]
    try:
        summary_chunks = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary_chunks[0]['summary_text']
    except Exception as e:
        return f"❌ Summarization failed: {str(e)}"

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_best_chunk(text, question):
    chunks = chunk_text(text)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    best_chunk_index = int(np.argmax(scores))
    return chunks[best_chunk_index]

def cohere_answer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    try:
        response = co.generate(
            model='command-light',
            prompt=prompt,
            max_tokens=150,
            temperature=0.5
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"❌ Cohere API Error: {str(e)}"

def generate_questions(text, num=3):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    sampled = random.sample(sentences, min(num, len(sentences)))
    questions = [f"What does the following mean: '{s}?'" for s in sampled]
    return questions, sampled

def evaluate_similarity(text1, text2):
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)[0].item()
    return round(score * 100, 2)

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_txt(file):
    return file.read().decode("utf-8")

# Session state
if 'ai_accuracy' not in st.session_state:
    st.session_state.ai_accuracy = None
if 'ai_answer' not in st.session_state:
    st.session_state.ai_answer = None
if 'challenge_questions' not in st.session_state:
    st.session_state.challenge_questions = None
if 'user_scores' not in st.session_state:
    st.session_state.user_scores = {}
if 'challenge_accuracy' not in st.session_state:
    st.session_state.challenge_accuracy = None

# Page config
st.set_page_config(page_title="GenAI Assistant", layout="wide")

# CSS and animation
st.markdown("""
<style>
body { margin: 0; overflow-x: hidden; }
#stars {
    background: black url('https://raw.githubusercontent.com/VincentGarreau/particles.js/master/demo/media/starfield.png') repeat top center;
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    display: block; z-index: -1;
    animation: moveStars 100s linear infinite;
    opacity: 0.15;
}
@keyframes moveStars {
    from { background-position: 0 0; }
    to { background-position: 10000px 5000px; }
}
.title-style {
    font-size: 3.2em; font-weight: 800; text-align: center;
    background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle-style {
    text-align: center; font-size: 1.2em; color: #bbb; margin-top: -10px;
}
.features-box {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    font-size: 1.05em;
    line-height: 1.7;
    color: #ffffff;
    margin-top: 20px;
    box-shadow: 0 0 15px rgba(0,0,0,0.3);
}
.content-box {
    background: rgba(0, 0, 0, 0.5);
    padding: 30px;
    border-radius: 15px;
    margin-top: 30px;
    margin-bottom: 30px;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
}
</style>
<div id="stars"></div>
""", unsafe_allow_html=True)

# Sidebar: AI & Challenge Accuracy
if st.session_state.ai_accuracy is not None:
    st.sidebar.markdown("### 📊 AI Answer Accuracy")
    st.sidebar.progress(st.session_state.ai_accuracy / 100)
    st.sidebar.metric("Ask Mode", f"{st.session_state.ai_accuracy}%")
    if st.session_state.ai_accuracy >= 75:
        st.sidebar.success("✅ Highly Relevant")
    elif st.session_state.ai_accuracy >= 50:
        st.sidebar.warning("⚠️ Somewhat Relevant")
    else:
        st.sidebar.error("❌ Not Relevant")

if st.session_state.challenge_accuracy is not None:
    st.sidebar.markdown("### 🧠 Challenge Mode Accuracy")
    st.sidebar.progress(st.session_state.challenge_accuracy / 100)
    st.sidebar.metric("User Avg", f"{st.session_state.challenge_accuracy}%")
    if st.session_state.challenge_accuracy >= 75:
        st.sidebar.success("✅ Great Job!")
    elif st.session_state.challenge_accuracy >= 50:
        st.sidebar.warning("⚠️ Can Improve")
    else:
        st.sidebar.error("❌ Needs Work")

# Title + Feature box
st.markdown('<div class="title-style">🤖 GenAI Smart Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Summarize, Ask, and Learn .</div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div class="features-box">
👋 Welcome! This GenAI-powered tool helps you:
<br><br>
✅ <b>Auto-Summarize</b> your documents instantly<br>
✅ <b>Ask intelligent questions</b> and get contextual answers<br>
✅ <b>Challenge yourself</b> with smart logic questions<br><br>
Just upload a PDF or TXT file below to begin 👇
</div>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.success("✅ File uploaded successfully!")

    text = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else read_txt(uploaded_file)

    st.markdown("---")
    st.subheader("📝 Auto Summary (≤150 words)")
    st.caption(f"🧾 Word Count: {len(text.split())}")
    summary = summarize_text(text)
    st.success(summary)

    st.markdown("---")
    st.subheader("📚 Extracted Document Content")
    with st.expander("📖 Click to view full document text"):
        st.write(text)

    st.markdown("---")
    st.subheader("❓ Ask Anything About the Document")
    user_question = st.text_input("Type your question here")

    if user_question:
        context = get_best_chunk(text, user_question)
        answer = cohere_answer(user_question, context)
        st.session_state.ai_answer = answer
        st.session_state.ai_accuracy = evaluate_similarity(answer, context)

        st.subheader("🧠 AI Answer")
        st.write(answer)

        st.markdown("##### 📌 Justification (Document Snippet)")
        st.code(context.strip()[:700] + ("..." if len(context) > 700 else ""), language="markdown")

        st.markdown("### 📊 Semantic Accuracy of AI Answer")
        st.metric("Accuracy", f"{st.session_state.ai_accuracy}%")
        if st.session_state.ai_accuracy >= 75:
            st.success("✅ Highly Relevant")
        elif st.session_state.ai_accuracy >= 50:
            st.warning("⚠️ Somewhat Relevant")
        else:
            st.error("❌ Not Relevant")

    st.markdown("---")
    st.subheader("🎯 Challenge Me Mode")
    if st.button("🧠 Generate 3 Logic Questions"):
        st.session_state.challenge_questions = generate_questions(text)
        st.session_state.user_scores = {}

    if st.session_state.challenge_questions:
        questions, answers = st.session_state.challenge_questions
        for i, (q, correct_ref) in enumerate(zip(questions, answers)):
            with st.expander(f"❓ Question {i+1}"):
                st.markdown(f"**Q{i+1}:** {q}")
                user_input = st.text_input(f"✏️ Your Answer to Q{i+1}", key=f"user_answer_{i}")
                if user_input:
                    user_score = evaluate_similarity(user_input, text)
                    st.session_state.user_scores[i] = user_score

                    st.markdown(f"**🧠 Your Answer Accuracy: {user_score}%**")
                    if user_score >= 75:
                        st.success("✅ Highly Relevant")
                    elif user_score >= 50:
                        st.warning("⚠️ Somewhat Relevant")
                    else:
                        st.error("❌ Not Relevant")

        if st.session_state.user_scores:
            all_scores = list(st.session_state.user_scores.values())
            average_score = round(sum(all_scores) / len(all_scores), 2)
            st.session_state.challenge_accuracy = average_score
            st.markdown("### 📈 Overall User Answer Accuracy")
            st.metric("Average Accuracy", f"{average_score}%")

    st.markdown('</div>', unsafe_allow_html=True)
