import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader 
from langchain_core.documents import Document 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Environment Variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. Page Configuration
st.set_page_config(page_title="Chat with Any Resume", page_icon="📄")
st.title("Resume AI Assistant 🤖")
st.write("Upload any PDF resume, and I'll become their personal assistant!")

# 3. The File Uploader (Placed in a sidebar for a clean UI)
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])

# 4. RAG Pipeline Setup 
# (We now pass the raw text into the cache. If you upload a new PDF, the text changes, and the cache auto-updates!)
@st.cache_resource(show_spinner=False)
def setup_rag_pipeline(pdf_text):
    # Wrap the raw string text into a format LangChain understands
    docs = [Document(page_content=pdf_text)]

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in a local FAISS vector database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Create the prompt template
    system_prompt = (
        "You are a helpful, professional assistant representing the candidate in the provided resume. "
        "Use the provided context to answer questions about their experience and skills. "
        "If the answer isn't in the context, say that you don't have that information. "
        "\n\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build the Retrieval Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# 5. Main App Logic: Only show chat if a file is uploaded
if uploaded_file is not None:
    
    # Read the PDF file directly from memory
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
        
    # Build the bot's brain using the extracted text
    with st.spinner("Analyzing resume..."):
        rag_chain = setup_rag_pipeline(raw_text)
    
    st.success("Resume loaded successfully! Start chatting below.")

    # Streamlit Chat Interface Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if user_input := st.chat_input("E.g., What is this person's core tech stack?"):
        # Add user message to UI and session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response from the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Scanning resume..."):
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                
        # Add assistant message to session state
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    # What the user sees before uploading a file
    st.info("👈 Please open the sidebar to the left and upload a PDF file to begin.")