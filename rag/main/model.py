from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


video_id="rBlCOLfMYfw"

transcript = YouTubeTranscriptApi().fetch(video_id).to_raw_data()

full_text = " ".join([t["text"] for t in transcript])


docs = [Document(page_content=full_text)]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstores=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./.chroma_db"
)

question="What is AI?"

retriever=vectorstores.as_retriever(search_type="mmr", search_kwargs={"k":3})
retrieved_docs=retriever.invoke(question)

context="".join(t.page_content for t in retrieved_docs)

prompt=PromptTemplate(
    template="Based on the context provided below {context} you have to answer the question {question} you are a conversational humble agent with 10 years of experience",
    input_variables=["context","question"]
)


model=ChatOpenAI()

parser=StrOutputParser()

chain=prompt|model|parser

response=chain.invoke({"context":context,"question":question})



print(response)
