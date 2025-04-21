import os
from mcp.server.fastmcp import FastMCP
# Ensure necessary imports are present if defining in a separate file
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from swagger_to_graphql_tool import swagger_to_graphql
from apollo_connector_decorator import decorate_with_apollo_connectors

# Define the path to your persisted vector store
# Make sure this matches the filename used in your creation script
PATH = "/Users/sergiobilello/Documents/repositories/mcp-demo"
PERSIST_PATH = "/Users/sergiobilello/Documents/repositories/mcp-demo/sklearn_gemini_apollo_vectorstore.parquet"
mcp = FastMCP("Hackathon-MCP-Server")

# Check if the vector store file exists before defining the tool,
# or handle potential errors inside the tool if it might not exist yet.
if not os.path.exists(PERSIST_PATH):
    print(f"Warning: Vector store file not found at {PERSIST_PATH}. The tool will likely fail.")
    # You could raise an error here, or let the tool fail later


@mcp.tool()
def apollo_query_tool(query: str):
    """
    Query the Apollo Connector documentation using a retriever backed by
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
            return "No relevant documents found in the Apollo Connector documentation for that query."

        formatted_context = "\n\n".join(
            [f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
        )
        return formatted_context

    except FileNotFoundError:
        return f"Error: Could not find the vector store file at {PERSIST_PATH}. Please ensure the index has been created."
    except Exception as e:
        # Catching other potential errors (e.g., API key issues, loading issues)
        print(f"An error occurred while querying the vector store: {e}")
        return f"Error: Could not query Apollo Connector documentation due to an internal issue: {e}"


@mcp.tool()
def swagger_to_graphql_tool(swagger_spec: str, save_output: bool = False):
    """
    Convert a Swagger/OpenAPI specification to GraphQL schema using @o/swagger-to-graphql.
    
    Args:
        swagger_spec (str): JSON string containing the Swagger/OpenAPI specification
        save_output (bool): Whether to save the output to a file (default: False)
        
    Returns:
        str: The generated GraphQL schema as a string and path to the saved file if save_output is True
    """
    try:
        # Create output path if saving is requested
        output_path = None
        if save_output:
            output_path = os.path.join(PATH, "generated_schema.graphql")
            
        # Call the swagger_to_graphql function
        graphql_schema = swagger_to_graphql(swagger_spec, output_path)
        
        # Return the result
        if save_output:
            return {
                "schema": graphql_schema,
                "saved_to": output_path
            }
        else:
            return {
                "schema": graphql_schema
            }
            
    except Exception as e:
        return {
            "error": str(e)
        }

@mcp.tool()
def decorate_graphql_with_apollo_connectors(graphql_schema: str, swagger_spec: str, save_output: bool = False):
    """
    Decorate a GraphQL schema with Apollo Connector directives based on the original Swagger spec.
    
    Args:
        graphql_schema (str): The GraphQL schema as a string
        swagger_spec (str): JSON string containing the original Swagger/OpenAPI specification
        save_output (bool): Whether to save the output to a file (default: False)
        
    Returns:
        dict: The decorated GraphQL schema and path to the saved file if save_output is True
    """
    try:
        # Use the apollo_query_tool to get Apollo Connector documentation
        connector_docs = apollo_query_tool("Apollo Connector directives and usage")
    
        # Decorate the schema with Apollo Connector directives
        decorated_schema = decorate_with_apollo_connectors(
            graphql_schema=graphql_schema,
            swagger_spec=swagger_spec,
            connector_docs=connector_docs
        )
        
        # Create output path if saving is requested
        output_path = None
        if save_output:
            output_path = os.path.join(PATH, "decorated_schema.graphql")
            with open(output_path, "w") as f:
                f.write(decorated_schema)
        
        # Return the result
        if save_output:
            return {
                "schema": decorated_schema,
                "saved_to": output_path
            }
        else:
            return {
                "schema": decorated_schema
            }
        
    except Exception as e:
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

#  #Example usage (if run in the same file/context for testing):
# if __name__ == '__main__':
#  #       Make sure GOOGLE_API_KEY is set in your environment
#     if os.path.exists(PERSIST_PATH):
#         test_query = "What is LangGraph?"
#         # Call the function directly, not using .invoke()
#         results = langgraph_query_tool(test_query)
#         print("\n=== Query Results ===")
#         print(results)
#     else:
#         print("Run the vector store creation script first.")