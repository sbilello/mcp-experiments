import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple, TypedDict # Import TypedDict correctly
import sys # Import sys for sys.exit

# Assuming graphql-core is installed: pip install graphql-core
try:
    from graphql import build_schema, validate, GraphQLError
except ImportError:
    print("Error: graphql-core library not found. Please install it: pip install graphql-core")
    sys.exit(1)

from langchain.tools import tool
from langgraph.graph import END, StateGraph

# Import the actual implementation functions directly
# Ensure these files (swagger_to_graphql_tool.py, apollo_connector_decorator.py)
# exist and contain the necessary functions.
try:
    from swagger_to_graphql_tool import swagger_to_graphql
    from apollo_connector_decorator import decorate_with_apollo_connectors
except ImportError as e:
    print(f"Error importing implementation functions: {e}")
    print("Please ensure swagger_to_graphql_tool.py and apollo_connector_decorator.py exist and are importable.")
    sys.exit(1)

# Define the state schema
class AgentState(TypedDict): # Inherit from TypedDict
    """State for the Swagger-to-GraphQL workflow."""
    # Input state
    swagger_spec: str # Made non-optional as it's the entry point

    # Processing state
    graphql_schema: Optional[str]
    decorated_schema: Optional[str]

    # Output state
    final_schema: Optional[str]
    output_path: Optional[str]

    # Error state
    error: Optional[str]

# Define the tools (nodes will call these)

@tool
def parse_swagger_spec(swagger_spec_str: str) -> Dict[str, Any]:
    """
    Parse and validate a Swagger/OpenAPI specification string.

    Args:
        swagger_spec_str: JSON string containing the Swagger/OpenAPI specification

    Returns:
        Dict containing the parsed specification dict or an error key.
    """
    print("--- PARSING SWAGGER ---")
    try:
        # Parse the JSON string
        parsed_spec = json.loads(swagger_spec_str)

        # Basic validation (can be expanded)
        required_fields = ["swagger", "info", "paths"] if "swagger" in parsed_spec else ["openapi", "info", "paths"]
        missing_fields = [field for field in required_fields if field not in parsed_spec]
        if missing_fields:
            return {"error": f"Invalid Swagger/OpenAPI spec: Missing required fields: {', '.join(missing_fields)}"}

        # Add more validation if needed (e.g., using openapi-spec-validator)

        print("Parsing successful.")
        return {"parsed_spec": parsed_spec} # Return the parsed dict
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"error": f"Error parsing Swagger spec: {str(e)}"}

@tool
def convert_to_graphql(parsed_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a parsed Swagger/OpenAPI specification dict to a GraphQL schema string
    using the underlying implementation function.

    Args:
        parsed_spec: Dictionary containing the parsed Swagger/OpenAPI specification.

    Returns:
        Dict containing the GraphQL schema string under the key 'graphql_schema'
        or an error message under the key 'error'.
    """
    print("--- CONVERTING TO GRAPHQL ---")
    if not parsed_spec:
         return {"error": "Parsed specification dictionary is missing in state."}
    try:
        # Call the underlying function, passing the dictionary.
        # Explicitly set output_path=None as saving is handled by a separate node.
        graphql_schema_str = swagger_to_graphql(
            swagger_spec=parsed_spec,
            output_path=None
        )

        if not graphql_schema_str or not isinstance(graphql_schema_str, str):
             raise ValueError("Conversion function did not return a valid schema string.")

        print("Conversion successful.")
        return {"graphql_schema": graphql_schema_str}
    except Exception as e:
        # Add traceback for better debugging if needed
        # import traceback
        # print(traceback.format_exc())
        return {"error": f"Error during swagger_to_graphql execution: {str(e)}"}

@tool
def decorate_with_connectors(graphql_schema: str, parsed_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decorate a GraphQL schema string with Apollo Connector directives.

    Args:
        graphql_schema: The GraphQL schema as a string.
        parsed_spec: Dictionary containing the parsed Swagger/OpenAPI specification.

    Returns:
        Dict containing the decorated schema string or an error key.
    """
    print("--- DECORATING SCHEMA ---")
    try:
        # Pass the schema string and parsed spec dict
        # Note: Ensure decorate_with_apollo_connectors accepts these arguments
        decorated_schema_str = decorate_with_apollo_connectors(
            graphql_schema=graphql_schema,
            # swagger_spec=json.dumps(parsed_spec) # Pass spec string if needed by decorator
            swagger_spec_dict=parsed_spec # Or pass the dict if needed
        )

        if not decorated_schema_str or not isinstance(decorated_schema_str, str):
             raise ValueError("Decoration function did not return a valid schema string.")

        print("Decoration successful.")
        return {"decorated_schema": decorated_schema_str}
    except Exception as e:
        return {"error": f"Error decorating schema: {str(e)}"}


@tool
def validate_graphql_schema(schema: str) -> Dict[str, Any]:
    """
    Validate a GraphQL schema string using graphql-core.

    Args:
        schema: The GraphQL schema as a string.

    Returns:
        Dict containing validation results (empty dict on success, error key on failure).
    """
    print("--- VALIDATING SCHEMA ---")
    try:
        if not schema or not isinstance(schema, str):
            return {"error": "Schema is empty or not a string"}

        # Attempt to build the schema
        gql_schema = build_schema(schema)

        # Validate the schema
        errors = validate(gql_schema)

        if errors:
            error_messages = [err.message for err in errors]
            return {"error": f"Invalid GraphQL schema: {'; '.join(error_messages)}"}

        # Optional: Add check for presence of expected directives if needed
        # if "@connector" not in schema: # Basic check
        #     return {"error": "Schema validation passed, but no @connector directives found."}

        print("Validation successful.")
        return {} # Return empty dict on success
    except GraphQLError as e:
         return {"error": f"GraphQL Syntax Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error validating schema: {str(e)}"}

@tool
def save_final_schema(schema: str, filename: str = "final_schema.graphql") -> Dict[str, Any]:
    """
    Save the final GraphQL schema string to a file.

    Args:
        schema: The GraphQL schema as a string.
        filename: The filename to save to (default: final_schema.graphql).

    Returns:
        Dict containing the path to the saved file or an error key.
    """
    print(f"--- SAVING SCHEMA TO {filename} ---")
    try:
        output_path = os.path.abspath(filename) # Use absolute path
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(schema)
        print(f"Schema saved successfully to {output_path}")
        return {"output_path": output_path}
    except Exception as e:
        return {"error": f"Error saving schema to {filename}: {str(e)}"}

# --- Define Graph Nodes ---

# Note: Nodes now only call the tools and return the *updates* to the state.

def parse_node(state: AgentState) -> Dict[str, Any]:
    """Node to parse and validate the Swagger specification."""
    result = parse_swagger_spec(state["swagger_spec"])
    # We store the original spec dict for later use if needed by decorator
    if "error" in result:
        return {"error": result["error"]}
    else:
        # Store the parsed dictionary, needed by the decorator potentially
        # Or pass state["swagger_spec"] string again if decorator needs string
        return {"parsed_spec": result["parsed_spec"]}


def convert_node(state: AgentState) -> Dict[str, Any]:
    """Node to convert the Swagger spec to a GraphQL schema."""
    # Pass the parsed spec if the tool expects a dict
    result = convert_to_graphql(state["parsed_spec"])
    # Or pass the original string if needed:
    # result = convert_to_graphql(state["swagger_spec"])
    if "error" in result:
        return {"error": result["error"]}
    else:
        return {"graphql_schema": result["schema"]}

def decorate_node(state: AgentState) -> Dict[str, Any]:
    """Node to decorate the GraphQL schema."""
    # Pass the generated schema and the original parsed spec dict
    result = decorate_with_connectors(state["graphql_schema"], state["parsed_spec"])
    # Or pass the original spec string if needed:
    # result = decorate_with_connectors(state["graphql_schema"], state["swagger_spec"])
    if "error" in result:
        return {"error": result["error"]}
    else:
        return {"decorated_schema": result["schema"]}

def validate_node(state: AgentState) -> Dict[str, Any]:
    """Node to validate the decorated GraphQL schema."""
    result = validate_graphql_schema(state["decorated_schema"])
    if "error" in result:
        return {"error": result["error"]}
    else:
        # Validation passed, promote decorated schema to final schema
        return {"final_schema": state["decorated_schema"]}

def save_node(state: AgentState) -> Dict[str, Any]:
    """Node to save the final GraphQL schema."""
    result = save_final_schema(state["final_schema"]) # Default filename
    # Could allow filename from initial state: save_final_schema(state["final_schema"], state.get("output_filename", "final_schema.graphql"))
    if "error" in result:
        return {"error": result["error"]}
    else:
        return {"output_path": result["path"]}

# --- Define Conditional Edges ---

def decide_next_step(state: AgentState) -> str:
    """Decide which node to run next based on the current state."""
    if state.get("error"):
        print(f"Error encountered: {state['error']}. Ending workflow.")
        return END # End workflow on any error

    if "parsed_spec" not in state:
        # Should not happen if parse is entrypoint, but good check
        return "parse"
    elif "graphql_schema" not in state:
        return "convert"
    elif "decorated_schema" not in state:
        return "decorate"
    elif "final_schema" not in state:
        # If decorated schema exists, next step is validation
        return "validate"
    elif "output_path" not in state:
        # If final schema exists (validation passed), next step is saving
        return "save"
    else:
        # All steps completed successfully
        print("Workflow completed successfully.")
        return END

# --- Create the Graph ---

def create_workflow() -> StateGraph:
    """Create the workflow graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("parse", parse_node)
    workflow.add_node("convert", convert_node)
    workflow.add_node("decorate", decorate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("save", save_node)

    # Set the entry point
    workflow.set_entry_point("parse")

    # Add conditional edges
    # After each step, decide where to go next (or end if error)
    workflow.add_conditional_edges(
        "parse",
        decide_next_step,
    )
    workflow.add_conditional_edges(
        "convert",
        decide_next_step,
    )
    workflow.add_conditional_edges(
        "decorate",
        decide_next_step,
    )
    workflow.add_conditional_edges(
        "validate",
        decide_next_step,
    )
    workflow.add_conditional_edges(
        "save",
        decide_next_step, # Will decide to END here if successful
    )

    return workflow

# Function to run the workflow
def run_swagger_to_graphql_agent(swagger_spec_str: str) -> Dict[str, Any]:
    """
    Run the Swagger-to-GraphQL workflow on a Swagger specification string.

    Args:
        swagger_spec_str: JSON string containing the Swagger/OpenAPI specification.

    Returns:
        Dict containing the final state after execution.
    """
    workflow = create_workflow()
    app = workflow.compile() # Compile the graph into a runnable application

    # Define the initial state, ensuring all optional fields start as None
    initial_state = AgentState(
        swagger_spec=swagger_spec_str,
        parsed_spec=None, # Add parsed_spec here
        graphql_schema=None,
        decorated_schema=None,
        final_schema=None,
        output_path=None,
        error=None
    )

    print("--- STARTING WORKFLOW ---")
    # Invoke the graph with the initial state
    # Use stream for potentially seeing intermediate steps, or invoke for final result
    # final_state_result = app.invoke(initial_state, {"recursion_limit": 10}) # Add recursion limit

    # Streaming execution to see step results (optional)
    final_state_result = None
    for step_result in app.stream(initial_state, {"recursion_limit": 15}):
         # stream() yields dictionaries where keys are node names
         node_name = list(step_result.keys())[0]
         node_output = step_result[node_name]
         print(f"\n--- Completed Step: {node_name} ---")
         print(f"Output State Keys: {list(node_output.keys())}")
         if node_output.get("error"):
              print(f"Error in step {node_name}: {node_output['error']}")
         final_state_result = node_output # Keep track of the last state


    print("--- WORKFLOW FINISHED ---")
    return final_state_result if final_state_result else initial_state # Return last state or initial if stream was empty

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <swagger_spec_file.json>") # Recommend JSON for easy loading
        sys.exit(1)

    swagger_file = sys.argv[1]
    if not os.path.exists(swagger_file):
        print(f"Error: Input file not found: {swagger_file}")
        sys.exit(1)

    try:
        # Read the Swagger spec from a file
        with open(swagger_file, "r", encoding="utf-8") as f:
            swagger_spec_content = f.read()
            # Pre-validate if it's JSON before passing to agent
            try:
                 json.loads(swagger_spec_content)
            except json.JSONDecodeError as json_err:
                 print(f"Error: Input file '{swagger_file}' is not valid JSON: {json_err}")
                 sys.exit(1)

        # Run the agent
        result_state = run_swagger_to_graphql_agent(swagger_spec_content)

        # Print the results
        print("\n--- FINAL RESULT ---")
        if result_state.get("error"):
            print(f"Workflow failed with error: {result_state['error']}")
        elif result_state.get("output_path"):
            print(f"Successfully converted Swagger spec.")
            print(f"Final GraphQL schema saved to: {result_state['output_path']}")

            # Print a sample of the schema
            if result_state.get("final_schema"):
                 schema_sample = result_state["final_schema"][:500] + "..." if len(result_state["final_schema"]) > 500 else result_state["final_schema"]
                 print(f"\nFinal schema sample:\n{schema_sample}")
            else:
                 print("Final schema data not found in result state.")
        else:
            print("Workflow finished, but no output path was generated. Final state:")
            # Print relevant parts of the final state for debugging
            print(f"  Error: {result_state.get('error')}")
            print(f"  GraphQL Schema Generated: {'Yes' if result_state.get('graphql_schema') else 'No'}")
            print(f"  Schema Decorated: {'Yes' if result_state.get('decorated_schema') else 'No'}")
            print(f"  Schema Validated (Final): {'Yes' if result_state.get('final_schema') else 'No'}")

    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)