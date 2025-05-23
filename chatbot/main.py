import os
from typing import List, Tuple, Dict # Dict for potential future use with type="messages"

# Set TOKENIZERS_PARALLELISM to false to avoid warnings from HuggingFace tokenizers
# This should be done before importing transformers or related HuggingFace libraries.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI # Updated import
import gradio as gr
import logging

# Attempt to load environment variables from .env file
load_dotenv()

# --- Configuration ---
PDF_PATH = os.getenv("PDF_PATH", "../docs/ai-ml.pdf") # Load from .env or default
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = 0.1
SEARCH_KWARGS = {"k": 3}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Global Variables for Core Logic ---
data_loaded = False
conversational_chain = None
pdf_status_message = "" # To store status of PDF loading and chain init

# --- Initialize Core Logic ---
try:
    if not OPENAI_API_KEY:
        pdf_status_message = "Warning: OPENAI_API_KEY not set. LLM features disabled."
        logging.warning(pdf_status_message)
    else:
        logging.info("OpenAI API Key found.")
        pdf_status_message = "OpenAI API Key found."


    logging.info(f"Attempting to load PDF from: {PDF_PATH}")
    if not os.path.exists(PDF_PATH):
        error_msg = f"Error: PDF file not found at {PDF_PATH}."
        pdf_status_message += f" {error_msg}"
        logging.error(error_msg)
    else:
        loader = UnstructuredPDFLoader(PDF_PATH)
        data = loader.load()
        if not data:
            error_msg = f"Error: No data loaded from PDF: {PDF_PATH}."
            pdf_status_message += f" {error_msg}"
            logging.error(error_msg)
        else:
            data_loaded = True
            success_msg = f"PDF '{os.path.basename(PDF_PATH)}' loaded."
            pdf_status_message += f" {success_msg}"
            logging.info(success_msg)

            if OPENAI_API_KEY: # Proceed only if API key and data are available
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                logging.info("HuggingFace embeddings model loaded.")
                
                db = Chroma.from_documents(data, embeddings)
                logging.info("Chroma vector store created.")
                
                llm = ChatOpenAI(
                    temperature=LLM_TEMPERATURE, openai_api_key=OPENAI_API_KEY
                )
                logging.info("ChatOpenAI LLM initialized.")
                
                conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm, retriever=db.as_retriever(search_kwargs=SEARCH_KWARGS)
                )
                chain_msg = "Conversational chain ready."
                pdf_status_message += f" {chain_msg}"
                logging.info(chain_msg)
            elif not OPENAI_API_KEY and data_loaded: # Data loaded but no API key
                 pdf_status_message += " Chain not created (OpenAI API Key missing)."


except Exception as e:
    error_detail = f"An error occurred during initialization: {e}"
    pdf_status_message = error_detail # Overwrite status with critical error
    logging.error(error_detail, exc_info=True)


# --- Gradio Functions ---
# Gradio's default chatbot history format is List[List[str, str]]
GradioChatHistoryType = List[List[str]]

def generate_response_for_chain( # Renamed from your original generate_response
    input_text: str, history_from_gradio: GradioChatHistoryType
) -> str:
    """
    Generates a response using the conversational chain.
    Converts Gradio's history (List[List[str,str]]) to List[Tuple[str,str]] for the chain.
    """
    if not conversational_chain:
        logging.warning(
            "Conversational chain not available for generating response."
        )
        return "Sorry, the document processing system is not available (chain not initialized)."
    
    history_for_langchain: List[Tuple[str, str]] = []
    for turn in history_from_gradio:
        if isinstance(turn, list) and len(turn) == 2:
            history_for_langchain.append((str(turn[0]), str(turn[1])))
        else:
            logging.warning(f"Skipping malformed turn in chat history: {turn}")
            
    try:
        logging.info(f"Invoking chain with question: {input_text}")
        logging.debug(f"History for chain: {history_for_langchain}")
        result = conversational_chain.invoke(
            {"question": input_text, "chat_history": history_for_langchain}
        )
        logging.info("Chain invocation successful.")
        return result["answer"]
    except Exception as e:
        logging.error(
            f"Error during conversational_chain.invoke: {e}", exc_info=True
        )
        return "Sorry, I encountered an error while generating a response. Please check the application logs."

# Renamed to 'chat' to match your UI block
def chat(
    message: str, chat_history: GradioChatHistoryType
) -> Tuple[str, GradioChatHistoryType]:
    """Handles a new chat message and updates history for Gradio."""
    if not message.strip(): 
        return "", chat_history

    # Using the renamed generate_response_for_chain
    response_text = generate_response_for_chain(message, chat_history) 
    chat_history.append([message, response_text]) 
    return "", chat_history 

# Renamed to 'initialize_chat' to match your UI block
def initialize_chat() -> Tuple[str, GradioChatHistoryType]:
    """Clears the chat input and history for Gradio."""
    return "", []

# --- Gradio UI (Reverted to your simpler structure) ---
with gr.Blocks() as app: # No custom theme applied, uses Gradio default
    gr.Markdown("# Chat with your PDF powered by Langchain & OpenAI")
    gr.Markdown(f"*{pdf_status_message}*") # Displaying the status

    chatbot = gr.Chatbot(label="Chat History") # 'chatbot' matches your UI
    msg = gr.Textbox(label="Your Message", placeholder="Type your question here...") # 'msg' matches
    
    # Using 'clear_button' as a variable name, but ClearButton component itself
    clear = gr.ClearButton([chatbot, msg], value="Clear Chat") 

    # Event handlers using the renamed functions 'chat' and 'initialize_chat'
    # The submit button is implicitly created by msg.submit if not defined separately
    # but defining it explicitly as in your example is also fine.
    # For consistency with your provided UI, I'll assume msg.submit is the primary way.
    
    # If you want an explicit send button as in your example:
    with gr.Row(): # This was in your original example, re-adding for structure
        # This submit_button is not strictly necessary if msg.submit is used,
        # but including it if you prefer an explicit button.
        # If you only want msg.submit, you can remove this explicit button.
        submit_button = gr.Button("Send") 

    submit_button.click(chat, [msg, chatbot], [msg, chatbot])
    msg.submit(chat, [msg, chatbot], [msg, chatbot]) 
    
    # ClearButton's click is handled by the component itself when components are passed.
    # If you used initialize_chat, it would be:
    # clear.click(initialize_chat, inputs=[], outputs=[msg, chatbot])
    # But ClearButton([chatbot, msg]) is more direct.

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        logging.warning(
            "CRITICAL: OPENAI_API_KEY environment variable not set. "
            "The application will have limited or no LLM functionality."
        )
    if not data_loaded:
        logging.warning(
            f"CRITICAL: No data was loaded from PDF: {PDF_PATH}. "
            "The chatbot will not have document context."
        )
    if not conversational_chain and data_loaded and OPENAI_API_KEY:
        logging.warning(
            "CRITICAL: Conversational chain was not initialized despite data and API key. "
            "Chat functionality will be impaired. Check previous logs for errors."
        )
    
    app.launch()
