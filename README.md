# 🎬 YouTube Insight
## End-to-End AI-Powered RAG Video Intelligence System

Turn any YouTube video into an interactive AI knowledge base using Retrieval-Augmented Generation (RAG).

This project demonstrates a complete production-style LLM pipeline including transcript ingestion, chunking, embedding generation, vector storage, semantic retrieval, context-grounded generation, web verification, and text-to-speech.

---

# 🚀 Demo Capabilities

- 📝 Generate structured AI summaries
- 💬 Ask contextual questions about the video
- 🔍 Perform semantic retrieval using embeddings
- 🔊 Convert summary to speech
- 🌐 Fact-check claims via web search
- 🧠 Context-grounded answers (anti-hallucination design)
- 💾 Persistent vector database per video

---


---

# 🧠 System Design Explanation

## 1️⃣ Data Ingestion Layer
- Extract transcript using YouTubeTranscriptApi
- Combine transcript segments into a single corpus

## 2️⃣ Preprocessing Layer
- Wrap transcript into LangChain Document objects
- Split into overlapping chunks (1000 chars, 200 overlap)
- Overlap preserves context across boundaries

## 3️⃣ Embedding Layer
- Convert chunks into dense vectors
- Model: `text-embedding-3-small`
- Each chunk becomes a semantic representation

## 4️⃣ Vector Database Layer
- Store embeddings in Chroma
- Persist directory: `./.chroma_db/{video_id}`
- Enables fast similarity search

## 5️⃣ Retrieval Layer
- Use MMR (Max Marginal Relevance)
- Returns top 4 semantically diverse chunks
- Prevents repetitive context

## 6️⃣ Generation Layer
- Prompt template enforces:
  - Use only retrieved context
  - Avoid hallucination
- Model: ChatOpenAI
- Low temperature for factual accuracy

## 7️⃣ Augmentation Layer
- DuckDuckGo web search for fact-checking
- OpenAI TTS for audio summaries

---


