# Rag Application

This is a Chainlit-based application that allows users to ask questions about child's story books and receive detailed, child-safe answers. The application uses the LangChain library to interact with the OpenAI API and retrieve relevant information from a set of PDF documents.

## Components

1. **API Key Handling**: The code starts by loading the OpenAI API key from an environment variable using the `dotenv` library.

2. **OpenAI Embedding Model**: An OpenAI Embedding model is created using the `OpenAIEmbeddings` class from the `langchain_openai` library.

3. **PDF Document Loading**: The `main()` function is annotated with `@cl.on_chat_start`, which means it will be executed when the Chainlit application starts. This function loads PDF documents from a folder named "pdfs/" and stores the text content in a list.

4. **Text Splitting**: The text content of the PDF documents is split into smaller chunks using the `CharacterTextSplitter` class from the `langchain.text_splitter` module. This is done to improve the performance of the search and retrieval process.

5. **FAISS Vector Store**: The chunked text is used to create a FAISS vector store using the `FAISS.from_documents()` function from the `langchain_community.vectorstores` module. This vector store will be used to perform similarity searches on the user's queries.

6. **Chat Model**: A `ChatOpenAI` model is created using the `langchain_openai` library. The temperature of the model is adjusted to 0.3 to generate more accurate responses. Lower temperatures can be used for less creative responses.

7. **LLM Chain**: An `LLMChain` is created using the `PromptTemplate` class from the `langchain_core.prompts` module and the `ConversationBufferMemory` class from the `langchain.memory` module. This chain combines the chat model, a custom prompt template, and conversation memory to generate the final response.
   - The `PromptTemplate` defines the initial prompt, which includes placeholders for the relevant document content, the chat history, and the user's input.
   - The `ConversationBufferMemory` is used to maintain the conversation context, allowing the language model to provide responses that are coherent and relevant to the previous prompts and answers.

8. **Retrieval QA Function**: The `get_response_from_query()` function takes a user's query, the FAISS vector store, and the LLM chain, and returns the generated response and the relevant documents.

9. **Chainlit App**: The `app()` function is annotated with `@cl.on_message`, which means it will be executed whenever a user sends a message to the Chainlit application. This function retrieves the necessary components from the user session and calls the `get_response_from_query()` function to generate the response, which is then sent back to the user.

## Approach

The main approach used in this application is to combine the power of LangChain and Chainlit to build a conversational AI assistant that can answer questions about child's story books. The key steps are:

1. Load and process the PDF documents to create a searchable vector store.
2. Define a custom prompt template that instructs the language model to provide detailed, child-safe answers based on the available information and the context of the conversation.
3. Use the `LLMChain` with the `ConversationBufferMemory` to maintain the context of the conversation and generate responses that are coherent and relevant to the previous prompts and answers.
4. Leverage the Chainlit framework to create a user-friendly chat interface and manage the application state across multiple users.

The application is designed to be scalable and reusable, with the FAISS vector store, chat model, and LLM chain stored in the user session for efficient retrieval and response generation.

## Requirements

The requirements.txt file contains the necessary Python packages for this application:

```
openai
python-dotenv
langchain-openai
langchain-community
chainlit
```

You can install these dependencies using pip:

```
pip install -r requirements.txt
```

Put the pdfs in the pdfs folder and you openai api key in .env file

## To run the code 

```
chainlit run main.py -w
```
