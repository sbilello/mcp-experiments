# --- Imports ---
import os
import logging
from mcp.server.fastmcp import FastMCP # Your MCP framework
from langchain_ollama import OllamaEmbeddings # Corrected import
from langchain_community.vectorstores import SKLearnVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) # Get directory of the script
OLLAMA_VECTORSTORE_FILENAME = "sklearn_ollama_vectorstore.parquet" # Ollama store
RAW_DOCS_FILENAME = "llms_full.txt"
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Ollama embedding model
RETRIEVER_K = 3 # Number of documents to retrieve

PERSIST_PATH = os.path.join(BASE_PATH, OLLAMA_VECTORSTORE_FILENAME)
DOC_PATH = os.path.join(BASE_PATH, RAW_DOCS_FILENAME)
# Optional: Specify Ollama base URL if not default localhost:11434
# OLLAMA_BASE_URL = "http://localhost:11434"

# --- Prerequisites Check ---
def check_prerequisites():
    """Checks for necessary files."""
    ready = True
    # Check for essential files
    if not os.path.exists(PERSIST_PATH):
        logging.error(f"CRITICAL: Ollama vector store file not found at {PERSIST_PATH}")
        logging.error("Please run the Ollama version of 'create_langgraph_index.py' first.")
        ready = False
    else:
        logging.info(f"Ollama vector store file found: {PERSIST_PATH}")

    if not os.path.exists(DOC_PATH):
        logging.error(f"CRITICAL: Documentation file not found at {DOC_PATH}")
        ready = False
    else:
        logging.info(f"Documentation file found: {DOC_PATH}")

    # Add a check for Ollama server reachability if desired (e.g., using requests)

    return ready

# --- MCP Server Setup ---
mcp = FastMCP("LangGraph-Docs-MCP-Server-Ollama") # Updated name
logging.info(f"MCP Server '{mcp.name}' initialized.")

# --- MCP Tool Definition (Using Ollama Embeddings) ---
@mcp.tool()
def langgraph_query_tool(query: str) -> str:
    """
    Query the LangGraph documentation using a retriever backed by
    a persisted SKLearnVectorStore with Ollama Embeddings.

    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A string containing the formatted retrieved documents,
             or an error message if the store cannot be loaded or queried.
    """
    logging.info(f"Tool 'langgraph_query_tool' called with query: '{query}'")

    if not os.path.exists(PERSIST_PATH):
        error_msg = f"Error: Could not find the Ollama vector store file at {PERSIST_PATH}. Please ensure the index has been created."
        logging.error(error_msg)
        return error_msg

    try:
        # Initialize the Ollama embedding function
        logging.info(f"Initializing Ollama Embeddings (Model: {EMBEDDING_MODEL_NAME})...")
        # Add base_url if needed: embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

        # Load the persisted vector store
        logging.info(f"Loading vector store from: {PERSIST_PATH}")
        vectorstore = SKLearnVectorStore(
            embedding=embeddings,
            persist_path=PERSIST_PATH,
            serializer="parquet"
        )
        logging.info("Vector store loaded successfully.")

        # Create a retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        logging.info(f"Retriever created (k={RETRIEVER_K}).")

        # Invoke the retriever
        logging.info("Invoking retriever...")
        relevant_docs = retriever.invoke(query)
        logging.info(f"Retrieved {len(relevant_docs)} relevant documents.")

        # Format the results
        if not relevant_docs:
            logging.info("No relevant documents found for the query.")
            return "No relevant documents found in the LangGraph documentation for that query."

        # Include source metadata
        formatted_context = "\n\n".join(
            [f"==DOCUMENT {i+1}==\nSource: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}"
             for i, doc in enumerate(relevant_docs)]
        )
        logging.info("Tool execution successful.")
        return formatted_context

    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        error_msg = f"Error: FileNotFoundError during vector store loading: {PERSIST_PATH}"
        logging.exception(error_msg)
        return f"Error: Could not load the vector store file at {PERSIST_PATH}."
    except Exception as e:
        error_msg = f"An error occurred while executing langgraph_query_tool: {e}"
        logging.exception(error_msg)
        # Provide a generic error message, check logs for details
        return f"Error: Could not query LangGraph documentation due to an internal issue (check Ollama server and model '{EMBEDDING_MODEL_NAME}')."

# --- MCP Resource Definition (Unchanged Logic) ---
@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Provides the full content of the LangGraph documentation file (llms_full.txt).
    """
    logging.info(f"Resource 'docs://langgraph/full' requested. Reading from: {DOC_PATH}")
    if not os.path.exists(DOC_PATH):
        error_msg = f"Error: Documentation file not found at {DOC_PATH}"
        logging.error(error_msg)
        return error_msg

    try:
        with open(DOC_PATH, 'r', encoding='utf-8') as file:
            content = file.read()
            logging.info(f"Successfully read {len(content)} characters from documentation file.")
            return content
    except Exception as e:
        error_msg = f"Error reading documentation file ({DOC_PATH}): {e}"
        logging.exception(error_msg)
        return f"Error reading documentation file: {str(e)}"

# --- Server Execution ---
if __name__ == "__main__":
    logging.info("--- Initializing MCP server (Ollama Version) ---")
    print("--- Ollama MCP Server ---")
    print("Before running:")
    print("1. Ensure Ollama server is running.")
    print(f"2. Ensure the embedding model '{EMBEDDING_MODEL_NAME}' is pulled: `ollama pull {EMBEDDING_MODEL_NAME}`")
    print(f"3. Ensure the vector store '{OLLAMA_VECTORSTORE_FILENAME}' exists (created by the Ollama index script).")
    print(f"4. Ensure the raw docs file '{RAW_DOCS_FILENAME}' exists.")
    print("5. Ensure required Python packages are installed:")
    print("   pip install langchain-community langchain-ollama scikit-learn pandas pyarrow <your-mcp-framework>")
    print("-" * 30)

    if check_prerequisites():
        logging.info("Prerequisites met. Starting server...")
        print("Server starting...")
        try:
            # Start the MCP server (adjust transport as needed)
            mcp.run(transport='stdio')
            logging.info("MCP server stopped.")
        except Exception as e:
            logging.critical(f"MCP server failed to run: {e}", exc_info=True)
            print(f"ERROR: MCP server failed to run: {e}")
    else:
        logging.critical("Prerequisites not met. Server cannot start.")
        print("ERROR: Prerequisites not met. Server cannot start. Check logs.")

