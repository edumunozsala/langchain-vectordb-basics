from langchain.document_loaders import ApifyDatasetLoader
from langchain.utilities import ApifyWrapper
from langchain.document_loaders.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


import streamlit as st

import io
import re
import sys
from typing import Any, Callable
import os


from dotenv import load_dotenv

# Read the URL 
def read_url(url):
    apify = ApifyWrapper()
    # Create the Actor
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": url}]},
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"] if dataset_item["text"] else "No content available",
            metadata={
                "source": dataset_item["url"],
                "title": dataset_item["metadata"]["title"]
            }
        ),
    )
    # Load the documents
    docs = loader.load()
    
    return docs
# Split the documents
def split_documents(docs, chunk_size, overlap):
     
    # we split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len
    )
    docs_split = text_splitter.split_documents(docs)
    
    return docs_split

#Create the embeddings
def create_embeddings(docs):
    # Create the embeddings
    embeddings = CohereEmbeddings(model = "embed-english-v2.0")

    username = "edumunozsala" # replace with your username from app.activeloop.ai
    db_id = 'rag-cohere-rerank' 
    # Delete the database if it already exists
    DeepLake.force_delete_by_path(f"hub://{username}/{db_id}")
    # Create the database
    dbs = DeepLake(dataset_path=f"hub://{username}/{db_id}", embedding_function=embeddings)
    # Add the documents
    dbs.add_documents(docs)
    
    return dbs
    
@st.cache_resource()
def data_lake():
    embeddings = CohereEmbeddings(model = "embed-english-v2.0")
    username = "edumunozsala" # replace with your username from app.activeloop.ai
    db_id = 'rag-cohere-rerank' 

    dbs = DeepLake(
        dataset_path="hub://{username}/{db_id}", 
        read_only=True, 
        embedding_function=embeddings
        )
    retriever = dbs.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    compressor = CohereRerank(
        model = 'rerank-english-v2.0',
        top_n=5
        )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    return dbs, compression_retriever, retriever

@st.cache_resource()
def memory():
    memory=ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True, 
        output_key='answer'
        )
    return memory

import streamlit as st
import io
import re
import sys
from typing import Any, Callable

def capture_and_display_output(func: Callable[..., Any], args, **kwargs) -> Any:
    # Capture the standard output
    original_stdout = sys.stdout
    sys.stdout = output_catcher = io.StringIO()

    # Run the given function and capture its output
    response = func(args, **kwargs)

    # Reset the standard output to its original value
    sys.stdout = original_stdout

    # Clean the captured output
    output_text = output_catcher.getvalue()
    clean_text = re.sub(r"\x1b[.?[@-~]", "", output_text)

    # Custom CSS for the response box
    st.markdown("""
    <style>
        .response-value {
            border: 2px solid #6c757d;
            border-radius: 5px;
            padding: 20px;
            background-color: #f8f9fa;
            color: #3d3d3d;
            font-size: 20px;  # Change this value to adjust the text size
            font-family: monospace;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create an expander titled "See Verbose"
    with st.expander("See Langchain Thought Process"):
        # Display the cleaned text in Streamlit as code
        st.code(clean_text)

    return response

def chat_ui(qa):
    # Accept user input
    if prompt := st.chat_input(
        "Ask me questions: How can I retrieve data from Deep Lake in Langchain?"
    ):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Load the memory variables, which include the chat history
            memory_variables = memory.load_memory_variables({})

            # Predict the AI's response in the conversation
            with st.spinner("Searching course material"):
                response = capture_and_display_output(
                    qa, ({"question": prompt, "chat_history": memory_variables})
                )

            # Display chat response
            full_response += response["answer"]
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            #Display top 2 retrieved sources
            source = response["source_documents"][0].metadata
            source2 = response["source_documents"][1].metadata
            with st.expander("See Resources"):
                st.write(f"Title: {source['title'].split('·')[0].strip()}")
                st.write(f"Source: {source['source']}")
                st.write(f"Relevance to Query: {source['relevance_score'] * 100}%")
                st.write(f"Title: {source2['title'].split('·')[0].strip()}")
                st.write(f"Source: {source2['source']}")
                st.write(f"Relevance to Query: {source2['relevance_score'] * 100}%")

        # Append message to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    
if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    # Read the URL
    url=""
    docs= read_url(url)
    # Split the documents
    docs= split_documents(docs, 1000, 20)
    #Create the embeddings
    dbs= create_embeddings(docs)
    # Reload the vector database
    dbs, compression_retriever, retriever = data_lake()
    # Create the memorymodule
    memory=memory()
    
    # Create the LLM
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=retriever, memory=memory, verbose= True, chain_type= 'stuff', return_source_documents=True
    )
    
    # Create a button to trigger the clearing of cache and session states
    if st.sidebar.button("Start a New Chat Interaction"):
        clear_cache_and_session()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Run function passing the ConversationalRetrievalChain
    chat_ui(qa)


    
    
        
