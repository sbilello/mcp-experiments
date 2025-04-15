# --- Imports ---
import re
import os
import logging
import tiktoken
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # Corrected import
from langchain_community.vectorstores import SKLearnVectorStore
import pandas  # Required by SKLearnVectorStore with parquet
import pyarrow # Required by SKLearnVectorStore with parquet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
EMBEDDING_MODEL_NAME = "nomic-embed-text"  # Ollama embedding model
OLLAMA_VECTORSTORE_FILENAME = "sklearn_ollama_vectorstore.parquet"
RAW_DOCS_FILENAME = "llms_full.txt"
CHUNK_SIZE = 1024  # Using the previously discussed size
CHUNK_OVERLAP = 150 # Using the previously discussed overlap
TIKTOKEN_MODEL = "cl100k_base" # Tiktoken model for splitting calculation
LANGGRAPH_DOC_URLS = [
    "https://langchain-ai.github.io/langgraph/concepts/",
    "https://langchain-ai.github.io/langgraph/how-tos/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/",
]
# Optional: Specify Ollama base URL if not default localhost:11434
# OLLAMA_BASE_URL = "http://localhost:11434"

# --- count_tokens function ---
def count_tokens(text: str, model: str = TIKTOKEN_MODEL) -> int:
    """Counts tokens using tiktoken (local, approximate)."""
    try:
        encoder = tiktoken.get_encoding(model)
        return len(encoder.encode(text))
    except Exception as e:
        logging.warning(f"Tiktoken counting failed for model {model}. Error: {e}. Falling back to char count / 4.")
        # Fallback: estimate based on characters (very rough)
        return len(text) // 4

# --- bs4_extractor function ---
def bs4_extractor(html: str) -> str:
    """Extracts main content text from LangGraph documentation HTML."""
    try:
        soup = BeautifulSoup(html, "lxml")
        main_content = soup.find("article", class_="md-content__inner")
        if main_content:
            # Use newline separator and strip whitespace for cleaner text
            content = main_content.get_text(separator="\n", strip=True)
        else:
            logging.warning("Could not find 'article.md-content__inner', falling back to full body text.")
            body = soup.find("body")
            content = body.get_text(separator="\n", strip=True) if body else ""

        # Clean up excessive consecutive newlines
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        return content
    except Exception as e:
        logging.error(f"Error during HTML extraction: {e}", exc_info=True)
        return "" # Return empty string on error

# --- load_langgraph_docs function ---
def load_langgraph_docs(urls: list) -> tuple[list, list[int]]:
    """Loads documents recursively from a list of URLs."""
    logging.info("Loading LangGraph documentation...")
    docs = []
    for url in urls:
        logging.info(f"Loading from URL: {url}")
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=5,
            extractor=bs4_extractor,
            prevent_outside=True,
            use_async=True,
            timeout=60,
            check_response_status=True, # Raise error for bad responses
        )
        try:
            # lazy_load returns an iterator, convert to list
            loaded_docs_iter = loader.lazy_load()
            docs.extend(list(loaded_docs_iter))
        except Exception as e:
            logging.error(f"Error loading URL {url}: {e}", exc_info=True)

    logging.info(f"Loaded {len(docs)} documents from LangGraph documentation.")
    if docs:
        logging.info("Sources of loaded documents:")
        for i, doc in enumerate(docs):
            logging.info(f"  {i+1}. {doc.metadata.get('source', 'Unknown URL')}")

    # Count total tokens in documents using tiktoken
    total_tokens = 0
    tokens_per_doc = []
    logging.info("Calculating token counts using tiktoken...")
    for doc in docs:
        doc_tokens = count_tokens(doc.page_content)
        total_tokens += doc_tokens
        tokens_per_doc.append(doc_tokens)
    logging.info(f"Total tokens (tiktoken approx.) in loaded documents: {total_tokens}")

    return docs, tokens_per_doc

# --- save_llms_full function ---
def save_llms_full(documents: list, output_filename: str = RAW_DOCS_FILENAME):
    """Saves the full text content of loaded documents to a file."""
    logging.info(f"Saving full document text to {output_filename}...")
    output_path = os.path.join(os.getcwd(), output_filename)
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            for i, doc in enumerate(documents):
                source = doc.metadata.get('source', 'Unknown URL')
                f.write(f"DOCUMENT {i+1}\n")
                f.write(f"SOURCE: {source}\n")
                f.write("CONTENT:\n")
                f.write(doc.page_content)
                f.write("\n\n" + "="*80 + "\n\n")
        logging.info(f"Documents concatenated into {output_path}")
    except Exception as e:
        logging.error(f"Error saving documents to {output_path}: {e}", exc_info=True)

# --- split_documents function ---
def split_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """Splits documents into smaller chunks using RecursiveCharacterTextSplitter."""
    if not documents:
        logging.warning("No documents provided to split.")
        return []

    logging.info(f"Splitting documents (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=TIKTOKEN_MODEL, # Specify model for tiktoken
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Keep separators default for general text
    )
    try:
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Created {len(split_docs)} chunks.")

        # Count total tokens in split documents using tiktoken
        total_tokens = sum(count_tokens(doc.page_content) for doc in split_docs)
        logging.info(f"Total tokens (tiktoken approx.) in split documents: {total_tokens}")

        return split_docs
    except Exception as e:
        logging.error(f"Error splitting documents: {e}", exc_info=True)
        return []

# --- create_vectorstore function ---
def create_vectorstore(splits: list, embedding_model: str = EMBEDDING_MODEL_NAME, persist_filename: str = OLLAMA_VECTORSTORE_FILENAME) -> SKLearnVectorStore | None:
    """
    Creates and persists a vector store using SKLearnVectorStore and Ollama Embeddings.
    Requires Ollama server to be running with the specified embedding model.
    """
    if not splits:
        logging.warning("No document splits provided. Cannot create vector store.")
        return None

    persist_path = os.path.join(os.getcwd(), persist_filename)
    logging.info(f"Creating SKLearnVectorStore with Ollama Embeddings (Model: {embedding_model})...")
    logging.info("Ensure Ollama server is running and the model is available.")

    try:
        # Initialize Ollama Embeddings using the corrected import
        # Add base_url if needed: embeddings = OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_BASE_URL)
        embeddings = OllamaEmbeddings(model=embedding_model)

        # Perform a small test embedding to catch connection errors early
        logging.info("Testing Ollama embedding connection...")
        _ = embeddings.embed_query("test query")
        logging.info("OllamaEmbeddings initialized and tested successfully.")

    except Exception as e:
        logging.error(f"Error initializing or testing OllamaEmbeddings: {e}", exc_info=True)
        logging.error(f"Please ensure the Ollama server is running, the model '{embedding_model}' is pulled (`ollama pull {embedding_model}`),")
        logging.error("and the 'langchain-ollama' package is installed correctly (`pip install langchain-ollama`).")
        return None # Return None if embeddings cannot be initialized

    # Create vector store from documents using SKLearn
    try:
        logging.info(f"Attempting to create and persist vector store at: {persist_path}")
        vectorstore = SKLearnVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_path=persist_path,
            serializer="parquet", # Use parquet for efficiency
        )
        logging.info(f"SKLearnVectorStore created and persisted successfully to {persist_path}")
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating or persisting SKLearnVectorStore: {e}", exc_info=True)
        logging.error("Ensure 'pandas' and 'pyarrow' are installed (`pip install pandas pyarrow`).")
        return None

# --- Main execution flow ---
def main():
    """Main function to orchestrate the document processing and vector store creation."""
    logging.info("--- Starting LangGraph Documentation Processing (Ollama Version) ---")

    # Load the documents
    loaded_documents, _ = load_langgraph_docs(LANGGRAPH_DOC_URLS)

    if not loaded_documents:
        logging.error("No documents were loaded. Exiting.")
        return # Exit if loading failed

    # Save the raw documents to a file
    save_llms_full(loaded_documents)

    # Split the documents
    split_docs = split_documents(loaded_documents)

    # Create the vector store (only if splits were created)
    if split_docs:
        vectorstore = create_vectorstore(split_docs)

        if vectorstore:
             logging.info("--- Vector store created and persisted successfully using Ollama Embeddings. ---")
             # Optional: Add an example query here for verification
             try:
                 results = vectorstore.similarity_search("What is StateGraph?")
                 logging.info(f"Example query returned {len(results)} results.")
                 if results:
                     logging.info(f"Top result snippet: {results[0].page_content[:200]}...")
             except Exception as e:
                 logging.error(f"Error running example query: {e}")
        else:
             logging.error("--- Failed to create vector store. ---")
    else:
        logging.warning("--- No document splits were created. Skipping vector store creation. ---")

    logging.info("--- Processing complete. ---")

if __name__ == "__main__":
    print("--- Ollama LangGraph Index Creator ---")
    print("Before running:")
    print("1. Ensure Ollama server is running.")
    print(f"2. Ensure the embedding model '{EMBEDDING_MODEL_NAME}' is pulled: `ollama pull {EMBEDDING_MODEL_NAME}`")
    print("3. Ensure required Python packages are installed:")
    print("   pip install langchain langchain-community langchain-ollama tiktoken beautifulsoup4 lxml scikit-learn pandas pyarrow")
    print("-" * 30)

    main()
