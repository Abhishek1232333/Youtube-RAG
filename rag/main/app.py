import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_icon="🎬",
    page_title="YouTube Insight - AI RAG Assistant",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Gradient Title */
    .stHeading h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 2rem !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }

    /* Card-like containers */
    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* Custom Info Box */
    .summary-box {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        color: #e2e8f0;
    }

    .chat-bubble {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* Input adjustments */
    .stTextInput input {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Functions ---
def extract_video_id(url):
    pattern = r'(?:v=|\/|be\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=80)
    st.title("YouTube RAG")
    st.markdown("### Powering Video Intelligence")
    st.write("Extract insights, summarize content, and ask questions to any YouTube video using AI.")
    
    st.divider()
    url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    process_btn = st.button("🚀 Process Video")
    
    if process_btn and url_input:
        video_id = extract_video_id(url_input)
        if video_id:
            try:
                with st.status("Analyzing Video Content...") as status:
                    st.write("Fetching transcript...")
                    transcript = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
                    full_text = " ".join([t["text"] for t in transcript])
                    
                    st.write("Chunking data...")
                    docs = [Document(page_content=full_text)]
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(docs)
                    
                    st.write("Generating embeddings & updating knowledge base...")
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    vectorstores = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=f"./.chroma_db/{video_id}"
                    )
                    
                    st.session_state.retriever = vectorstores.as_retriever(search_type="mmr", search_kwargs={"k": 4})
                    st.session_state.full_text = full_text
                    st.session_state.processed_url = url_input
                    st.session_state.chat_history = []
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.success("Video ready for interaction!")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
        else:
            st.error("Please enter a valid YouTube URL.")

# --- Main Interface ---
if "retriever" not in st.session_state:
    st.title("Welcome to YouTube Insight")
    st.info("👈 Enter a YouTube URL in the sidebar to begin your interactive experience.")
    
    # Showcase features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📝 Smart Summary")
        st.write("Get a high-level overview and key takeaways in seconds.")
    with col2:
        st.markdown("### 💬 Interactive Chat")
        st.write("Ask specific questions and get answers directly from the video content.")
    with col3:
        st.markdown("### 🎯 Precision RAG")
        st.write("Powered by OpenAI Embeddings and ChromaDB for accurate retrieval.")

else:
    header_col, action_col = st.columns([2, 1])
    with header_col:
        st.title("Video Intelligence Dashboard")
    
    # Display Video
    st.video(st.session_state.processed_url)
    
    tab1, tab2, tab3 = st.tabs(["✨ Summary & Insights", "💬 Ask Questions", "🔍 Web Verify"])
    
    with tab1:
        st.subheader("Video Takeaways")
        if st.button("🪄 Generate Summary", key="sum_btn"):
            with st.spinner("Analyzing themes..."):
                summary_prompt = PromptTemplate(
                    template="""You are a world-class AI learning assistant. 
                    Based on the following transcript, provide a professional summary and 5 crucial key points to learn.
                    Transcript: {full_text}
                    Format: Use Markdown headers and clean bullet points.""",
                    input_variables=["full_text"]
                )
                model = ChatOpenAI(temperature=0.3)
                parser = StrOutputParser()
                chain = summary_prompt | model | parser
                summary = chain.invoke({"full_text": st.session_state.full_text})
                st.session_state.summary_text = summary
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

        # Audio Summary Section
        if "summary_text" in st.session_state:
            st.write("---")
            if st.button("🔊 Play Audio Summary"):
                from openai import OpenAI
                client = OpenAI()
                with st.spinner("Preparing audio..."):
                    try:
                        response = client.audio.speech.create(
                            model="tts-1",
                            voice="alloy",
                            input=st.session_state.summary_text[:4000]
                        )
                        st.audio(response.content, format="audio/mpeg")
                    except Exception as e:
                        st.error(f"TTS Error: {e}")

    with tab2:
        st.subheader("Chat with Video")
        st.info("Directly ask questions based on the video transcript.")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask anything about the video...", key="rag_chat"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Searching video context..."):
                    qa_prompt = PromptTemplate(
                        template="""Context: {context}
                        Question: {question}
                        Answer as a professional and humble expert using ONLY the video context provided. 
                        If the answer isn't in the video, politely state that the video doesn't cover this topic.""",
                        input_variables=["context", "question"]
                    )
                    model = ChatOpenAI(temperature=0.2)
                    parser = StrOutputParser()
                    
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
                    
                    chain = qa_prompt | model | parser
                    response = chain.invoke({"context": context_text, "question": prompt})
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    with tab3:
        st.subheader("External Fact-Check & Search")
        st.markdown("Verify the information from the video or search for related topics on the web using **DuckDuckGo**.")
        
        verify_col1, verify_col2 = st.columns([2, 1])
        
        with verify_col2:
            st.write("### Quick Actions")
            if "summary_text" in st.session_state:
                if st.button("🔍 Verify Summary Claims"):
                    from langchain_community.tools import DuckDuckGoSearchRun
                    search = DuckDuckGoSearchRun()
                    with st.spinner("Searching web..."):
                        query = f"Fact check claims: {st.session_state.summary_text[:150]}"
                        try:
                            results = search.run(query)
                            st.session_state.verify_results = results
                        except Exception as e:
                            st.error(f"Search error: {e}")
            else:
                st.warning("Generate a summary first to use live verification.")

        with verify_col1:
            verify_input = st.text_input("Verify a specific statement or claim:", placeholder="e.g. Is [Topic] actually true according to other sources?")
            if st.button("🔎 Search Web", use_container_width=True):
                if verify_input:
                    from langchain_community.tools import DuckDuckGoSearchRun
                    search = DuckDuckGoSearchRun()
                    with st.spinner(f"Verifying '{verify_input}'..."):
                        try:
                            results = search.run(verify_input)
                            st.session_state.verify_results = results
                        except Exception as e:
                            st.error(f"Search failed: {e}")
            
            if "verify_results" in st.session_state:
                st.markdown("---")
                st.markdown("### 🌐 Web Verification Results")
                st.info(st.session_state.verify_results)
