import os
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
import chainlit as cl
from langchain.memory import ConversationBufferMemory


# Load API key from .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create OpenAI Embedding model
embeddings = OpenAIEmbeddings()

# Initialize the Chainlit application
@cl.on_chat_start
def main():
    # Load documents from PDF folder
    pdf_folder = "pdfs/"
    documents = []
    for pdf_file in Path(pdf_folder).glob("*.pdf"):
        loader = PDFMinerLoader(pdf_folder + str(pdf_file.name))
        text = loader.load()
        documents.extend(text)

    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create FAISS vector store
    db = FAISS.from_documents(texts, embeddings)

    # Create the chat model adujusting the temperature to 0.3 for more accurate responses can reduce the temperature for less creative response
    chat = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")

    # Store the FAISS vector store and chat model in the user session for reusability
    cl.user_session.set("db", db)
    cl.user_session.set("chat", chat)

    # Create the LLM chain with a custom prompt
    prompt = PromptTemplate(
        input_variables=["docs" ,"chat_history","human_input"],
        template=("""
        You are a helpful assistant that can answer questions about child's story books.
        here is the chat history so far:
        {chat_history}
        Answer the following question based on the context of chat history if it is related: {human_input}
        By searching the following text: {docs}
        Only use the factual information from the text to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose, detailed, and child-safe.
        """)
    )
    memory=ConversationBufferMemory(
    memory_key="chat_history",
    input_key="human_input"
    )
    chain = LLMChain(llm=chat, prompt=prompt,memory=memory,verbose=False)
    cl.user_session.set("llm_chain", chain)

# Define the retrieval QA function
def get_response_from_query(db, query, chain, k=20):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    response = chain.predict(docs=docs_page_content,human_input=query)
    return response, docs

# Define the Chainlit app
@cl.on_message
async def app(message: cl.Message):
    if message.content:
        db = cl.user_session.get("db")
        chain = cl.user_session.get("llm_chain")
        result, docs = get_response_from_query(db, message.content, chain)
        await cl.Message(content=result).send()