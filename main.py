import re, os
import tiktoken # Keep for the splitter if needed, or remove if only using Google counter
import google.generativeai as genai # Import Google library

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_anthropic import ChatAnthropic # This wasn't used but keep if needed elsewhere
from langchain_community.vectorstores import SKLearnVectorStore

# Configure Google API Key (place this after imports)
# Ensure GOOGLE_API_KEY environment variable is set
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    print("Google API Key configured.")
else:
    print("Warning: GOOGLE_API_KEY environment variable not set. Google token counting and embeddings may fail.")

# --- count_tokens function REPLACED ---
def count_tokens(text, model="models/embedding-001"):
    """
    Count the number of tokens in the text using Google's tokenizer via API.

    Args:
        text (str): The text to count tokens for.
        model (str): The Google model name (default: models/embedding-001).

    Returns:
        int: Number of tokens according to the specified Google model.
             Returns 0 if the API is not configured or counting fails.
    """
    if not genai.is_configured():
         print("Warning: Google API not configured. Cannot count tokens.")
         return 0

    try:
        # Use the Google Generative AI library's count_tokens method
        # Note: This might make an API call. Consider performance implications.
        response = genai.count_tokens(model=model, contents=[text]) # Pass text as list
        return response.total_tokens
    except Exception as e:
        print(f"Warning: Google token counting failed for model {model}. Error: {e}")
        # Fallback or return 0/None if counting fails
        return 0 # Return 0 to avoid breaking calculations

# --- tiktoken based counter (optional, keep if splitter needs it) ---
# def count_tokens_tiktoken(text, model="cl100k_base"):
#     """Counts tokens using tiktoken (local, approximate for non-OpenAI)."""
#     try:
#         encoder = tiktoken.get_encoding(model)
#         return len(encoder.encode(text))
#     except Exception as e:
#         print(f"Warning: Tiktoken counting failed. Error: {e}")
#         return 0

# --- bs4_extractor function remains the same ---
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Target the main article content for LangGraph documentation
    main_content = soup.find("article", class_="md-content__inner")

    # If found, use that, otherwise fall back to the whole document
    content = main_content.get_text() if main_content else soup.text

    # Clean up whitespace
    content = re.sub(r"\n\n+", "\n\n", content).strip()

    return content

# --- load_langgraph_docs function MODIFIED ---
def load_langgraph_docs():
    """
    Load LangGraph documentation from the official website.

    This function:
    1. Uses RecursiveUrlLoader to fetch pages from the LangGraph website
    2. Counts the total documents and tokens loaded

    Returns:
        list: A list of Document objects containing the loaded content
        list: A list of tokens per document
    """
    print("Loading LangGraph documentation...")

    # Load the documentation
    urls = ["https://langchain-ai.github.io/langgraph/concepts/",
     "https://langchain-ai.github.io/langgraph/how-tos/",
     "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
     "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
     "https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/",
    ]

    docs = []
    for url in urls:

        loader = RecursiveUrlLoader(
            url,
            max_depth=5,
            extractor=bs4_extractor,
            prevent_outside=True, # Good practice to prevent crawling outside the target domain
            use_async=True, # Often faster for multiple URLs
            timeout=60, # Set a timeout
        )

        # Load documents using lazy loading (memory efficient)
        # Handle potential errors during loading
        try:
            docs_lazy = loader.lazy_load()
            # Load documents and track URLs
            for d in docs_lazy:
                docs.append(d)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")


    print(f"Loaded {len(docs)} documents from LangGraph documentation.")
    if docs: # Only print URLs if documents were loaded
        print("\nLoaded URLs:")
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.metadata.get('source', 'Unknown URL')}")

    # Count total tokens in documents using the updated count_tokens (Google's)
    total_tokens = 0
    tokens_per_doc = []
    print("Calculating token counts using Google API...")
    for doc in docs:
        doc_tokens = count_tokens(doc.page_content) # Use the updated function
        total_tokens += doc_tokens
        tokens_per_doc.append(doc_tokens)
    print(f"Total tokens (Google count) in loaded documents: {total_tokens}") # Updated label

    return docs, tokens_per_doc

# --- save_llms_full function remains the same ---
def save_llms_full(documents):
    """ Save the documents to a file """

    # Open the output file
    output_filename = "llms_full.txt"

    with open(output_filename, "w", encoding='utf-8') as f: # Added encoding
        # Write each document
        for i, doc in enumerate(documents):
            # Get the source (URL) from metadata
            source = doc.metadata.get('source', 'Unknown URL')

            # Write the document with proper formatting
            f.write(f"DOCUMENT {i+1}\n")
            f.write(f"SOURCE: {source}\n")
            f.write("CONTENT:\n")
            f.write(doc.page_content)
            f.write("\n\n" + "="*80 + "\n\n")

    print(f"Documents concatenated into {output_filename}")

# --- split_documents function MODIFIED ---
def split_documents(documents):
    """
    Split documents into smaller chunks.
    NOTE: RecursiveCharacterTextSplitter.from_tiktoken_encoder still uses tiktoken
          for the *splitting logic* itself (chunk size calculation).
          The reporting below will use the Google token counter.
    """
    print("Splitting documents (using tiktoken-based chunk size calculation)...")

    # This splitter *calculates chunk size* based on tiktoken approximation.
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} chunks.")

    # Count total tokens in split documents using the updated count_tokens (Google's)
    total_tokens = 0
    print("Calculating token counts for splits using Google API...")
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content) # Use the updated function

    print(f"Total tokens (Google count) in split documents: {total_tokens}") # Updated label

    return split_docs

# --- create_vectorstore function remains the same ---
def create_vectorstore(splits):
    """
    Create a vector store from document chunks using SKLearnVectorStore.

    This function:
    1. Initializes a Google embedding model to convert text into vector representations
    2. Creates a vector store from the document chunks

    Args:
        splits (list): List of split Document objects to embed

    Returns:
        SKLearnVectorStore: A vector store containing the embedded documents
    """
    print("Creating SKLearnVectorStore with Google Embeddings...")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        print(f"Error initializing GoogleGenerativeAIEmbeddings: {e}")
        print("Please ensure the 'langchain-google-genai' package is installed")
        print("and your GOOGLE_API_KEY environment variable is set correctly.")
        return None # Return None if embeddings cannot be initialized

    # Create vector store from documents using SKLearn
    persist_path = os.path.join(os.getcwd(), "sklearn_gemini_vectorstore.parquet") # Changed filename slightly
    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=persist_path   ,
        serializer="parquet",
    )
    print("SKLearnVectorStore created successfully.")

    vectorstore.persist()
    print(f"SKLearnVectorStore was persisted to {persist_path}")

    return vectorstore

# --- Main execution flow ---

# Before running, ensure you have installed the required package:
# pip install langchain-google-genai tiktoken beautifulsoup4 lxml langchain langchain-community scikit-learn langchain-anthropic pandas pyarrow # Added pandas/pyarrow for parquet

# Also, set your Google API Key as an environment variable:
# export GOOGLE_API_KEY="YOUR_API_KEY_HERE" # (Linux/macOS)
# set GOOGLE_API_KEY=YOUR_API_KEY_HERE # (Windows Command Prompt)
# $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE" # (Windows PowerShell)


# Load the documents
documents, tokens_per_doc = load_langgraph_docs()

if documents: # Only proceed if documents were loaded successfully
    # Save the documents to a file
    save_llms_full(documents)

    # Split the documents
    split_docs = split_documents(documents)

    # Create the vector store
    vectorstore = create_vectorstore(split_docs)

    if vectorstore:
        print("\nVector store created and persisted successfully using Google Embeddings.")
        # You can now use the 'vectorstore' object for similarity searches, etc.
        # Example: results = vectorstore.similarity_search("some query about LangGraph")
        # print(results)
    else:
        print("\nFailed to create vector store.")
else:
    print("\nNo documents were loaded. Skipping splitting and vector store creation.")

    # Load the documents
documents, tokens_per_doc = load_langgraph_docs()

# Save the documents to a file
save_llms_full(documents)

# Split the documents
split_docs = split_documents(documents)

# Create the vector store
vectorstore = create_vectorstore(split_docs)

