import os
from mcp.server.fastmcp import FastMCP
# Ensure necessary imports are present if defining in a separate file
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

# Define the path to your persisted vector store
# Make sure this matches the filename used in your creation script
PATH = os.getcwd()
PERSIST_PATH = os.path.join(os.getcwd(), "sklearn_gemini_vectorstore.parquet")
mcp = FastMCP("LangGraph-Docs-MCP-Server")

# Check if the vector store file exists before defining the tool,
# or handle potential errors inside the tool if it might not exist yet.
if not os.path.exists(PERSIST_PATH):
    print(f"Warning: Vector store file not found at {PERSIST_PATH}. The tool will likely fail.")
    # You could raise an error here, or let the tool fail later


@mcp.tool(name="LangGraph-Docs-Query")
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
            # No need to call from_documents here, loading is implicit when providing path
        )

        # Create a retriever from the loaded vector store
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Get top 3 results

        # Invoke the retriever
        relevant_docs = retriever.invoke(query)
        print(f"Retrieved {len(relevant_docs)} relevant documents for query: '{query}'")

        # Format the results
        if not relevant_docs:
            return "No relevant documents found in the LangGraph documentation for that query."

        formatted_context = "\n\n".join(
            [f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
        )
        return formatted_context

    except FileNotFoundError:
        return f"Error: Could not find the vector store file at {PERSIST_PATH}. Please ensure the index has been created."
    except Exception as e:
        # Catching other potential errors (e.g., API key issues, loading issues)
        print(f"An error occurred while querying the vector store: {e}")
        return f"Error: Could not query LangGraph documentation due to an internal issue: {e}"

# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Get all the LangGraph documentation. Returns the contents of the file llms_full.txt,
    which contains a curated set of LangGraph documentation (~300k tokens). This is useful
    for a comprehensive response to questions about LangGraph.

    Args: None

    Returns:
        str: The contents of the LangGraph documentation
    """

    # Local path to the LangGraph documentation
    doc_path = PATH + "/llms_full.txt"
    try:
        with open(doc_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

# Example usage (if run in the same file/context for testing):
# if __name__ == '__main__':
#     # Make sure GOOGLE_API_KEY is set in your environment
#     if os.path.exists(PERSIST_PATH):
#         test_query = "How do I add conditional edges in LangGraph?"
#         results = langgraph_query_tool.invoke({"query": test_query}) # Tool decorator handles input dict
#         print("\n=== Query Results ===")
#         print(results)
#     else:
#         print("Run the vector store creation script first.")
