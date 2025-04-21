import re, os
import tiktoken # Used for token counting and splitting logic
import google.generativeai as genai # Used for configuring API key for embeddings

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
import pandas # Required by SKLearnVectorStore with parquet
import pyarrow # Required by SKLearnVectorStore with parquet

# Configure Google API Key (place this after imports)
# Ensure GOOGLE_API_KEY environment variable is set
google_api_key_configured = False
if "GOOGLE_API_KEY" in os.environ:
    try:
        # Configure API key for GoogleGenerativeAIEmbeddings
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        google_api_key_configured = True
        print("Google API Key configured (for embeddings).")
    except Exception as e:
        print(f"Error configuring Google API: {e}")
else:
    print("Warning: GOOGLE_API_KEY environment variable not set. Google embeddings will fail.")

# --- count_tokens function REVERTED to use tiktoken ---
def count_tokens(text, model="cl100k_base"):
    """
    Counts tokens using tiktoken (local, approximate).
    Using cl100k_base as a general approximation.
    """
    try:
        encoder = tiktoken.get_encoding(model)
        return len(encoder.encode(text))
    except Exception as e:
        print(f"Warning: Tiktoken counting failed for model {model}. Error: {e}")
        return 0

# --- bs4_extractor function remains the same ---
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    main_content = soup.find("article", class_="md-content__inner")
    content = main_content.get_text() if main_content else soup.text
    content = re.sub(r"\n\n+", "\n\n", content).strip()
    return content

# --- load_langgraph_docs function MODIFIED ---
def load_langgraph_docs():
    """
    Load LangGraph documentation from the official website.
    Counts tokens using tiktoken.
    """
    print("Loading LangGraph documentation...")
    urls = ["https://langchain-ai.github.io/langgraph/concepts/",
     "https://langchain-ai.github.io/langgraph/how-tos/",
     "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
     "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
     "https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/",
    ]
    docs = []
    for url in urls:
        loader = RecursiveUrlLoader(
            url, max_depth=5, extractor=bs4_extractor,
            prevent_outside=True, use_async=True, timeout=60,
        )
        try:
            docs_lazy = loader.lazy_load()
            for d in docs_lazy:
                docs.append(d)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")

    print(f"Loaded {len(docs)} documents from LangGraph documentation.")
    if docs:
        print("\nLoaded URLs:")
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.metadata.get('source', 'Unknown URL')}")

    # Count total tokens in documents using tiktoken
    total_tokens = 0
    tokens_per_doc = []
    print("Calculating token counts using tiktoken...")
    for doc in docs:
        doc_tokens = count_tokens(doc.page_content) # Use tiktoken counter
        total_tokens += doc_tokens
        tokens_per_doc.append(doc_tokens)
    print(f"Total tokens (tiktoken count) in loaded documents: {total_tokens}")

    return docs, tokens_per_doc

# --- save_llms_full function remains the same ---
def save_llms_full(documents):
    """ Save the documents to a file """
    output_filename = "llms_full.txt"
    try:
        with open(output_filename, "w", encoding='utf-8') as f:
            for i, doc in enumerate(documents):
                source = doc.metadata.get('source', 'Unknown URL')
                f.write(f"DOCUMENT {i+1}\n")
                f.write(f"SOURCE: {source}\n")
                f.write("CONTENT:\n")
                f.write(doc.page_content)
                f.write("\n\n" + "="*80 + "\n\n")
        print(f"Documents concatenated into {output_filename}")
    except Exception as e:
        print(f"Error saving documents to {output_filename}: {e}")


# --- split_documents function MODIFIED ---
def split_documents(documents):
    """
    Split documents into smaller chunks using tiktoken-based splitter.
    Reports token counts using tiktoken.
    """
    print("Splitting documents (using tiktoken-based chunk size calculation)...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} chunks.")

    # Count total tokens in split documents using tiktoken
    total_tokens = 0
    print("Calculating token counts for splits using tiktoken...")
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content) # Use tiktoken counter
    print(f"Total tokens (tiktoken count) in split documents: {total_tokens}")

    return split_docs

# --- create_vectorstore function REVISED ---
def create_vectorstore(splits):
    """
    Create a vector store using SKLearnVectorStore and Google Embeddings.
    Requires GOOGLE_API_KEY to be configured.
    """
    print("Creating SKLearnVectorStore with Google Embeddings...")
    embedding_model_name = "models/embedding-001"

    if not google_api_key_configured:
         print("\nError: Google API Key not configured. Cannot create Google embeddings.")
         print("Please set the GOOGLE_API_KEY environment variable.")
         return None # Cannot proceed without API key

    try:
        # Initialize Google Embeddings (requires API key to be configured)
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
        # Perform a small test embedding to catch configuration/auth errors early
        _ = embeddings.embed_query("test")
        print("GoogleGenerativeAIEmbeddings initialized successfully.")

    except Exception as e:
        print(f"\nError initializing or testing GoogleGenerativeAIEmbeddings: {e}")
        print("Please ensure the 'langchain-google-genai' package is installed correctly,")
        print("your GOOGLE_API_KEY is valid, and the model name is correct.")
        return None # Return None if embeddings cannot be initialized

    # Create vector store from documents using SKLearn
    persist_path = os.path.join(os.getcwd(), "sklearn_gemini_vectorstore.parquet")
    try:
        print(f"Attempting to create vector store at: {persist_path}")
        vectorstore = SKLearnVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_path=persist_path,
            serializer="parquet",
        )
        print("SKLearnVectorStore created successfully.")
        vectorstore.persist()
        print(f"SKLearnVectorStore was persisted to {persist_path}")
        return vectorstore
    except Exception as e:
        print(f"\nError creating or persisting SKLearnVectorStore: {e}")
        print("Ensure 'pandas' and 'pyarrow' are installed.")
        # If the error is about file path, check permissions.
        return None

# --- Main execution flow ---

# Before running, ensure you have installed the required packages:
# pip install google-generativeai langchain-google-genai tiktoken beautifulsoup4 lxml langchain langchain-community scikit-learn pandas pyarrow

# Also, set your Google API Key as an environment variable:
# export GOOGLE_API_KEY="YOUR_API_KEY_HERE" # (Linux/macOS)
# set GOOGLE_API_KEY=YOUR_API_KEY_HERE # (Windows Command Prompt)
# $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE" # (Windows PowerShell)

print("--- Starting LangGraph Documentation Processing ---")

# Load the documents
loaded_documents, doc_tokens = load_langgraph_docs()

# Check if documents were loaded before proceeding
if loaded_documents:
    # Save the documents to a file
    save_llms_full(loaded_documents)

    # Split the documents
    split_docs = split_documents(loaded_documents)

    # Create the vector store (only if splits were created)
    if split_docs:
        vectorstore = create_vectorstore(split_docs)

        # if vectorstore:
        #     print("\n--- Vector store created and persisted successfully using Google Embeddings. ---")
        #     # Example usage:
        #     try:
        #         # Create retriever to get relevant documents (k=3 means return top 3 matches)
        #         retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        #         # Get relevant documents for the query
        #         query = "Do I need to use LangChain to use LangGraph? What's the difference?"    
        #         relevant_docs = retriever.invoke(query)
        #         print(f"Retrieved {len(relevant_docs)} relevant documents")

        #         for d in relevant_docs:
        #             print(d.metadata['source'])
        #             print(d.page_content[0:500])
        #             print("\n--------------------------------\n")
        #     except Exception as search_e:
        #          print(f"\nError during example similarity search: {search_e}")
        # else:
        #     print("\n--- Failed to create vector store. ---")
    else:
        print("\n--- No document splits were created. Skipping vector store creation. ---")
else:
    print("\n--- No documents were loaded. Exiting. ---")

print("\n--- Processing complete. ---")

