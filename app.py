from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from transformers import GPT2Tokenizer
from htmlTemplates import css, bot_template, user_template
import requests

# Load the GPT-2 tokenizer for accurate token counting
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define a function to count tokens accurately using GPT-2 tokenizer
def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# Adjust the chat history to fit within the token limit
def limit_chat_history_by_tokens(chat_history, max_tokens=3000):
    total_tokens = sum([count_tokens(message['content']) for message in chat_history])
    while total_tokens > max_tokens and len(chat_history) > 1:
        chat_history.pop(0)  # Remove the oldest message
        total_tokens = sum([count_tokens(message['content']) for message in chat_history])
    return chat_history

# Optional: Summarize old chat history if it's too long
def summarize_chat_history(chat_history):
    summarizer = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    old_history = "\n".join([message['content'] for message in chat_history[:3]])  # Just an example to get the last 3 messages
    summary = summarizer(f"Summarize the following conversation: {old_history}")
    return summary['choices'][0]['message']['content']

# Custom prompt template for follow-up question rephrasing
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# PDF text extraction function
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the text into smaller chunks with smaller token count
def get_chunks(raw_text, chunk_size=400):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=100, length_function=len)
    return text_splitter.split_text(raw_text)

# Convert the chunks into a vector store for efficient retrieval
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Adjusted get_conversationchain method
def get_conversationchain(vectorstore):
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0.2)
    
    # Initialize the memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    # Get the current chat history from the memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Limit chat history to the max token limit
    chat_history = limit_chat_history_by_tokens(chat_history)

    # Optionally, summarize if it's still too long
    if sum([count_tokens(message['content']) for message in chat_history]) > 3000:
        summarized_history = summarize_chat_history(chat_history)
        memory.save_context({"input": "", "output": summarized_history})  # Save summarized content in memory

    # Create and return the conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )
    return conversation_chain

# Handle the question and get a response from the conversation chain

import requests  # Import requests for handling HTTP errors like BadRequestError

def handle_question(question):
    try:
        response = st.session_state.conversation({'question': question})
        st.session_state.chat_history = response["chat_history"]
        
        # Check if the response is meaningful or empty
        if not response["chat_history"][-1].content.strip():
            # Display error message if the response is empty
            st.write(bot_template.replace("{{MSG}}", "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"), unsafe_allow_html=True)
        else:
            for i, msg in enumerate(st.session_state.chat_history):

                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

    except :
        # Handle the case where the context exceeds the model's limit
        
        error_message = "Sorry, there was an issue processing your request. Do you want to connect with a live agent?"
        
        # Display the error message
        st.write(bot_template.replace("{{MSG}}", error_message), unsafe_allow_html=True)

# Main function for setting up Streamlit UI and handling logic
def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Questions about your PDFs", page_icon=":books", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # UI for uploading PDFs and interacting with the conversation
    st.markdown("""
        <style>
            .top-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                background-color: #cfe2f3;
                border-bottom: 1px solid #3d85c6;
            }
            .center-box {
                display: flex;
                justify-content: center;
                margin-top: 50px;
            }
            .input-box {
                width: 60%;
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="top-bar"><h2>Ask Questions About Your Documents</h2></div>', unsafe_allow_html=True)

    # PDF upload and processing button
    uploaded_files = st.file_uploader("Upload your PDFs here:", accept_multiple_files=True, key="upload")
    if st.button("Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversationchain(vectorstore)

    # UI for asking questions
    st.markdown('<div class="center-box">', unsafe_allow_html=True)
    question = st.text_input("Ask a question from your document:", key="input", help="Type your question here!")
    st.markdown('</div>', unsafe_allow_html=True)

    if question:
        handle_question(question)

if __name__ == '__main__':
    main()
