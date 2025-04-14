import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI # Import Gemini Chat Model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

# --- Assume your langgraph_query_tool is defined as in the previous step ---
# Make sure the PERSIST_PATH and embedding model used here match tool definition
PERSIST_PATH = os.path.join(os.getcwd(), "sklearn_gemini_vectorstore.parquet")

# (Ensure the tool definition is available in this scope)
@tool
def langgraph_query_tool(query: str):
    """
    Query the LangGraph documentation using a retriever backed by
    a persisted SKLearnVectorStore with Google Embeddings.

    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A string containing the formatted retrieved documents,
             or an error message if the store cannot be loaded.
    """
    try:
        # Initialize the embedding function - MUST match the one used for creation
        # Ensure GOOGLE_API_KEY environment variable is set
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load the persisted vector store
        vectorstore = SKLearnVectorStore(
            embedding=embeddings,
            persist_path=PERSIST_PATH,
            serializer="parquet"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        print(f"Retrieved {len(relevant_docs)} relevant documents for query: '{query}'")

        if not relevant_docs:
            return "No relevant documents found in the LangGraph documentation for that query."

        formatted_context = "\n\n".join(
            [f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
        )
        return formatted_context

    except FileNotFoundError:
        return f"Error: Could not find the vector store file at {PERSIST_PATH}. Please ensure the index has been created."
    except Exception as e:
        print(f"An error occurred while querying the vector store: {e}")
        return f"Error: Could not query LangGraph documentation due to an internal issue: {e}"
# --- End of tool definition ---


# --- Main part using Gemini ---

# Before running:
# 1. Ensure GOOGLE_API_KEY environment variable is set
# 2. Ensure langchain-google-genai is installed: pip install langchain-google-genai
# 3. Ensure the vector store file (PERSIST_PATH) exists

# Initialize the Gemini LLM
# Use a model that supports tool calling, like gemini-1.5-flash, gemini-1.5-pro or gemini-pro
# convert_system_message_to_human=True is often helpful for Gemini compatibility
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", # Or "gemini-1.5-pro-latest", "gemini-pro"
    temperature=0,
    convert_system_message_to_human=True # Recommended for Gemini
    )

# Bind the tool to the LLM - THIS LINE REMAINS THE SAME
augmented_llm = llm.bind_tools([langgraph_query_tool])

# Define instructions and messages - THIS REMAINS THE SAME
instructions = """You are a helpful assistant that can answer questions about the LangGraph documentation.
Use the langgraph_query_tool for any questions about the documentation.
If the tool returns an error or no relevant documents, say you couldn't find the information.
If you don't know the answer and the tool wasn't helpful, say "I don't know"."""

messages = [
    {"role": "system", "content": instructions},
    {"role": "user", "content": "What is LangGraph?"} # Example query
    # {"role": "user", "content": "How do I handle cycles?"} # Another example
]

# Invoke the LLM with bound tools - THIS REMAINS THE SAME
# The LLM will decide whether to call the tool based on the prompt and its capabilities
ai_message = augmented_llm.invoke(messages)

# Pretty print the response (which might include tool calls and the final answer)
# Note: The output structure includes tool_calls if the LLM decided to use the tool
ai_message.pretty_print()

# To see just the final content from the AI:
# print("\n=== AI Final Response ===")
# print(ai_message.content)