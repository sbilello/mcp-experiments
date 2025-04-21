import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple, TypedDict # Import TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
import langgraph.checkpoint as checkpoint

# Import the MCP tools - using direct function calls instead of client
# We'll directly call the functions from our modules instead of using MCP client
from swagger_to_graphql_tool import swagger_to_graphql
from apollo_connector_decorator import decorate_with_apollo_connectors

# Define the state schema
class AgentState(TypedDict): # Inherit from TypedDict
    """State for the Swagger-to-GraphQL agent."""
    
    # Input state
    swagger_spec: Optional[str]
    
    # Processing state
    graphql_schema: Optional[str]
    decorated_schema: Optional[str]
    
    # Output state
    final_schema: Optional[str]
    output_path: Optional[str]
    
    # Error state
    error: Optional[str]
    
    # Human-in-the-loop state
    human_input: Optional[str]
    messages: List[Union[HumanMessage, AIMessage]]


# Define the LLM to use for the agent
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Define the agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert agent that converts Swagger/OpenAPI specifications to GraphQL schemas with Apollo Connector directives.
    
Your task is to:
1. Parse and validate the Swagger/OpenAPI specification
2. Convert it to a GraphQL schema
3. Decorate the schema with appropriate Apollo Connector directives
4. Validate the final schema

Use the available tools to accomplish these tasks. Be thorough and explain your reasoning.
"""),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}"),
])

# Define the tools for the agent

@tool
def parse_swagger_spec(swagger_spec: str) -> Dict[str, Any]:
    """
    Parse and validate a Swagger/OpenAPI specification.
    
    Args:
        swagger_spec: JSON string containing the Swagger/OpenAPI specification
        
    Returns:
        Dict containing the parsed specification or error
    """
    try:
        # Parse the JSON string
        parsed_spec = json.loads(swagger_spec)
        
        # Basic validation
        required_fields = ["swagger", "info", "paths"] if "swagger" in parsed_spec else ["openapi", "info", "paths"]
        for field in required_fields:
            if field not in parsed_spec:
                return {"error": f"Invalid Swagger/OpenAPI spec: Missing required field '{field}'"}
        
        return {"spec": parsed_spec}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Error parsing Swagger spec: {str(e)}"}

@tool
def convert_to_graphql(swagger_spec: str) -> Dict[str, Any]:
    """
    Convert a Swagger/OpenAPI specification to a GraphQL schema.
    
    Args:
        swagger_spec: JSON string containing the Swagger/OpenAPI specification
        
    Returns:
        Dict containing the GraphQL schema or error
    """
    try:
        # Call the function directly to convert Swagger to GraphQL
        graphql_schema = swagger_to_graphql(swagger_spec)
        
        # Return the result
        return {"schema": graphql_schema}
    except Exception as e:
        return {"error": f"Error converting to GraphQL: {str(e)}"}

@tool
def decorate_with_connectors(graphql_schema: str, swagger_spec: str) -> Dict[str, Any]:
    """
    Decorate a GraphQL schema with Apollo Connector directives.
    
    Args:
        graphql_schema: The GraphQL schema as a string
        swagger_spec: JSON string containing the Swagger/OpenAPI specification
        
    Returns:
        Dict containing the decorated schema or error
    """
    try:
        # Call the function directly to decorate the schema
        decorated_schema = decorate_with_apollo_connectors(
            graphql_schema=graphql_schema,
            swagger_spec=swagger_spec
        )
        
        # Return the result
        return {"schema": decorated_schema}
    except Exception as e:
        return {"error": f"Error decorating schema: {str(e)}"}

@tool
def save_final_schema(schema: str, filename: str = "final_schema.graphql") -> Dict[str, Any]:
    """
    Save the final GraphQL schema to a file.
    
    Args:
        schema: The GraphQL schema as a string
        filename: The filename to save to (default: final_schema.graphql)
        
    Returns:
        Dict containing the path to the saved file or error
    """
    try:
        # Save the schema to a file
        output_path = os.path.join(os.getcwd(), filename)
        with open(output_path, "w") as f:
            f.write(schema)
        
        return {"path": output_path}
    except Exception as e:
        return {"error": f"Error saving schema: {str(e)}"}

@tool
def validate_graphql_schema(schema: str) -> Dict[str, Any]:
    """
    Validate a GraphQL schema.
    
    Args:
        schema: The GraphQL schema as a string
        
    Returns:
        Dict containing validation results
    """
    # This is a simplified validation that just checks for basic syntax
    # In a real implementation, you would use a proper GraphQL validator
    try:
        # Check for basic GraphQL syntax
        if not schema or not isinstance(schema, str):
            return {"valid": False, "errors": ["Schema is empty or not a string"]}
        
        # Check for type definitions
        if "type " not in schema:
            return {"valid": False, "errors": ["No type definitions found in schema"]}
        
        # Check for Query type
        if "type Query" not in schema:
            return {"valid": False, "errors": ["No Query type defined in schema"]}
        
        # Check for connector directives
        if "@connector" not in schema:
            return {"valid": False, "errors": ["No @connector directives found in schema"]}
        
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}

# Define the agent's workflow

def agent_node(state: AgentState) -> Dict[str, Any]:
    """Process the current state and determine the next action."""
    
    # Initialize messages if empty
    if not state.messages:
        state.messages = [
            HumanMessage(content=f"I need to convert a Swagger/OpenAPI specification to a GraphQL schema with Apollo Connector directives. Here's the specification: {state.swagger_spec[:100]}...")
        ]
    
    # Get the last message
    last_message = state.messages[-1]
    
    # If the last message is from the human, process it
    if isinstance(last_message, HumanMessage):
        # Generate a response using the LLM
        response = llm.invoke([
            ("system", """You are an expert agent that converts Swagger/OpenAPI specifications to GraphQL schemas with Apollo Connector directives.
            
Your task is to:
1. Parse and validate the Swagger/OpenAPI specification
2. Convert it to a GraphQL schema
3. Decorate the schema with appropriate Apollo Connector directives
4. Validate the final schema

Use the available tools to accomplish these tasks. Be thorough and explain your reasoning.
"""),
            *[(m.type, m.content) for m in state.messages]
        ])
        
        # Add the response to the messages
        state.messages.append(AIMessage(content=response.content))
    
    # Return the updated state
    return {"messages": state.messages}

def decide_next_step(state: AgentState) -> str:
    """Decide what to do next based on the current state."""
    
    # If there's an error, go to human
    if state.error:
        return "human_in_the_loop"
    
    # If we don't have a GraphQL schema yet, go to convert
    if not state.graphql_schema:
        return "convert"
    
    # If we don't have a decorated schema yet, go to decorate
    if not state.decorated_schema:
        return "decorate"
    
    # If we have a decorated schema but no final schema, go to validate
    if not state.final_schema:
        return "validate"
    
    # If we have a final schema but no output path, go to save
    if not state.output_path:
        return "save"
    
    # If we have everything, we're done
    return END

def parse_node(state: AgentState) -> AgentState:
    """Parse and validate the Swagger specification."""
    
    # Parse the Swagger spec
    result = parse_swagger_spec(state.swagger_spec)
    
    # Check for errors
    if "error" in result:
        state.error = result["error"]
    
    return state

def convert_node(state: AgentState) -> AgentState:
    """Convert the Swagger spec to a GraphQL schema."""
    
    # Convert the Swagger spec to GraphQL
    result = convert_to_graphql(state.swagger_spec)
    
    # Check for errors
    if "error" in result:
        state.error = result["error"]
    else:
        state.graphql_schema = result["schema"]
    
    return state

def decorate_node(state: AgentState) -> AgentState:
    """Decorate the GraphQL schema with Apollo Connector directives."""
    
    # Decorate the GraphQL schema
    result = decorate_with_connectors(state.graphql_schema, state.swagger_spec)
    
    # Check for errors
    if "error" in result:
        state.error = result["error"]
    else:
        state.decorated_schema = result["schema"]
    
    return state

def validate_node(state: AgentState) -> AgentState:
    """Validate the decorated GraphQL schema."""
    
    # Validate the GraphQL schema
    result = validate_graphql_schema(state.decorated_schema)
    
    # Check for errors
    if not result["valid"]:
        state.error = f"Invalid GraphQL schema: {', '.join(result['errors'])}"
    else:
        state.final_schema = state.decorated_schema
    
    return state

def save_node(state: AgentState) -> AgentState:
    """Save the final GraphQL schema to a file."""
    
    # Save the final schema
    result = save_final_schema(state.final_schema)
    
    # Check for errors
    if "error" in result:
        state.error = result["error"]
    else:
        state.output_path = result["path"]
    
    return state

def human_in_the_loop_node(state: AgentState) -> Tuple[AgentState, str]:
    """Handle human interaction for error resolution."""
    
    # Add the error message to the messages
    state.messages.append(
        AIMessage(content=f"I encountered an error: {state.error}. Please provide guidance on how to proceed.")
    )
    
    # Return to the agent node
    return state, "agent"

# Create the graph
def create_workflow() -> StateGraph:
    """Create the workflow graph for the Swagger-to-GraphQL agent."""
    
    # Create a new graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("convert", convert_node)
    workflow.add_node("decorate", decorate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("save", save_node)
    workflow.add_node("human_in_the_loop", human_in_the_loop_node)
    
    # Add the edges
    workflow.add_edge("agent", decide_next_step)
    workflow.add_edge("parse", decide_next_step)
    workflow.add_edge("convert", decide_next_step)
    workflow.add_edge("decorate", decide_next_step)
    workflow.add_edge("validate", decide_next_step)
    workflow.add_edge("save", decide_next_step)
    workflow.add_edge("human_in_the_loop", "agent")
    
    # Set the entry point
    workflow.set_entry_point("parse")
    
    return workflow

# Function to run the agent
def run_swagger_to_graphql_agent(swagger_spec: str) -> Dict[str, Any]:
    """
    Run the Swagger-to-GraphQL agent on a Swagger specification.
    
    Args:
        swagger_spec: JSON string containing the Swagger/OpenAPI specification
        
    Returns:
        Dict containing the final GraphQL schema and output path
    """
    # Create the workflow
    workflow = create_workflow()
    
    # Create the initial state
    initial_state = AgentState(swagger_spec=swagger_spec)
    
    # Run the workflow
    final_state = workflow.invoke(initial_state)
    
    # Return the results
    return {
        "schema": final_state.final_schema,
        "output_path": final_state.output_path,
        "error": final_state.error,
        "messages": [m.content for m in final_state.messages]
    }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python swagger_to_graphql_agent.py <swagger_spec_file>")
        sys.exit(1)
    
    # Read the Swagger spec from a file
    with open(sys.argv[1], "r") as f:
        swagger_spec = f.read()
    
    # Run the agent
    result = run_swagger_to_graphql_agent(swagger_spec)
    
    # Print the results
    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print(f"Successfully converted Swagger spec to GraphQL schema with Apollo Connector directives.")
        print(f"Output saved to: {result['output_path']}")
        
        # Print a sample of the schema
        schema_sample = result["schema"][:500] + "..." if len(result["schema"]) > 500 else result["schema"]
        print(f"\nSchema sample:\n{schema_sample}")
