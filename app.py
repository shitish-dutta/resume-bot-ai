import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# =========================
# LOAD ENV
# =========================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Resume AI Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* =========================
   GLOBAL
========================= */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(99,102,241,0.08), transparent 25%),
        radial-gradient(circle at bottom right, rgba(56,189,248,0.08), transparent 25%),
        linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #ffffff 100%
        );

    color: #0f172a;
}

/* =========================
   REMOVE STREAMLIT ELEMENTS
========================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =========================
   MAIN CONTAINER
========================= */

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

/* =========================
   HERO SECTION
========================= */

.hero-title {
    font-size: 4rem;
    font-weight: 800;

    text-align: center;

    color: #312e81;

    margin-bottom: 10px;

    animation: fadeUp 0.8s ease;
}

.hero-subtitle {
    text-align: center;

    color: #64748b;

    font-size: 1.08rem;

    margin-bottom: 50px;

    animation: fadeUp 1.1s ease;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

/* =========================
   SIDEBAR
========================= */

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.72);

    border-right: 1px solid rgba(226,232,240,1);

    backdrop-filter: blur(14px);
}

/* =========================
   GLASS CARD
========================= */

.glass {
    background: rgba(255,255,255,0.62);

    border: 1px solid rgba(255,255,255,0.8);

    backdrop-filter: blur(12px);

    border-radius: 24px;

    padding: 28px;

    box-shadow:
        0 4px 20px rgba(15,23,42,0.05),
        0 1px 2px rgba(15,23,42,0.04);
}

/* =========================
   CHAT BUBBLES
========================= */

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.55);

    border: 1px solid rgba(226,232,240,1);

    border-radius: 20px;

    padding: 16px;

    margin-bottom: 14px;

    backdrop-filter: blur(8px);
}

/* =========================
   FILE UPLOADER
========================= */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.45);

    border: 1px dashed #cbd5e1;

    border-radius: 18px;

    padding: 12px;
}

/* =========================
   CHAT INPUT
========================= */

[data-testid="stChatInput"] {
    border-radius: 18px;
}

/* =========================
   BUTTONS
========================= */

.stButton > button {
    width: 100%;

    border-radius: 14px;

    border: none;

    height: 3em;

    background: linear-gradient(
        90deg,
        #6366f1,
        #8b5cf6
    );

    color: white;

    font-weight: 600;

    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 10px 22px rgba(99,102,241,0.22);
}

/* =========================
   SUCCESS / INFO BOXES
========================= */

.stSuccess {
    border-radius: 16px;
}

.stInfo {
    border-radius: 16px;
}

/* =========================
   SCROLLBAR
========================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #c7d2fe;
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero-title">
Resume AI Assistant
</div>

<div class="hero-subtitle">
Upload any resume and chat with an AI assistant trained on the candidate profile.
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.markdown("## 📄 Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload a PDF Resume",
        type=["pdf"]
    )

    st.markdown("---")

    st.markdown("""
### ✨ Features

- AI Resume Chat
- Skill Analysis
- Experience Insights
- Candidate Q&A
- Context-Aware Responses
""")

# =========================
# RAG PIPELINE
# =========================
@st.cache_resource(show_spinner=False)
def setup_rag_pipeline(pdf_text):

    docs = [Document(page_content=pdf_text)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )

    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    system_prompt = (
        "You are a professional AI assistant representing "
        "the candidate in the uploaded resume. "
        "Use the provided context to answer questions about "
        "their skills, experience, education, projects, and achievements. "
        "If the answer isn't available in the context, "
        "say that you don't have that information. "
        "\\n\\nContext:\\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    return rag_chain

# =========================
# MAIN LOGIC
# =========================
if uploaded_file is not None:

    pdf_reader = PdfReader(uploaded_file)

    raw_text = ""

    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    with st.spinner("Analyzing Resume..."):

        rag_chain = setup_rag_pipeline(raw_text)

    st.success("Resume Loaded Successfully!")

    # =========================
    # CHAT HISTORY
    # =========================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # =========================
    # USER INPUT
    # =========================
    if user_input := st.chat_input(
        "Ask anything about the candidate..."
    ):

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = rag_chain.invoke({
                    "input": user_input
                })

                answer = response["answer"]

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:

    st.markdown("""
    <div class="glass" style="
    text-align:center;
    padding:70px;
    margin-top:40px;
    ">

    <div style="
    font-size:80px;
    margin-bottom:20px;
    ">
    📄
    </div>

    <h2 style="
    color:#0f172a;
    margin-bottom:10px;
    ">
    Upload a Resume to Begin
    </h2>

    <p style="
    color:#64748b;
    font-size:1.05rem;
    line-height:1.8;
    max-width:650px;
    margin:auto;
    ">
    Your AI assistant will analyze the uploaded resume
    and answer questions about the candidate's
    skills, projects, experience, and achievements.
    </p>

    </div>
    """, unsafe_allow_html=True)
