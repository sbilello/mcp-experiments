import os
import json
import tempfile
import subprocess
from typing import Dict, Any, Optional, Union

def swagger_to_graphql(swagger_spec: Union[str, Dict[str, Any]], output_path: Optional[str] = None) -> str:
    """
    Convert a Swagger/OpenAPI specification to GraphQL schema using @o/swagger-to-graphql.
    If the npm package is not installed, it will generate a basic GraphQL schema.
    
    Args:
        swagger_spec: Either a JSON string or a dictionary containing the Swagger/OpenAPI spec
        output_path: Optional path to save the generated GraphQL schema (if None, only returns the schema)
        
    Returns:
        str: The generated GraphQL schema as a string
    """
    # Determine if input is a string (JSON) or dictionary
    if isinstance(swagger_spec, dict):
        swagger_json = swagger_spec
    else:
        try:
            swagger_json = json.loads(swagger_spec)
        except json.JSONDecodeError:
            # Assume it's a YAML file path or content
            # For YAML handling, we'd need to add PyYAML as a dependency
            raise ValueError("Input appears to be YAML, which is not supported yet. Please provide JSON.")
    
    # Check if npm is installed
    try:
        # Create a temporary directory to store the swagger spec
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the Swagger spec to a temporary file
            swagger_file_path = os.path.join(temp_dir, "swagger_spec.json")
            with open(swagger_file_path, "w") as f:
                json.dump(swagger_json, f)
            
            # Create a temporary file for the output
            output_file_path = os.path.join(temp_dir, "schema.graphql")
            
            try:
                # Run the swagger-to-graphql command
                cmd = f"npx @o/swagger-to-graphql --swagger={swagger_file_path} > {output_file_path}"
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True)
                
                # Read the generated GraphQL schema
                with open(output_file_path, "r") as f:
                    graphql_schema = f.read()
                
                # If output_path is provided, save the schema there as well
                if output_path:
                    with open(output_path, "w") as f:
                        f.write(graphql_schema)
                
                return graphql_schema
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to convert using npm package: {e.stderr.decode() if e.stderr else str(e)}")
                print("Falling back to basic GraphQL schema generation...")
                return _generate_basic_graphql_schema(swagger_json, output_path)
    except Exception as e:
        print(f"Warning: Error using npm package: {str(e)}")
        print("Falling back to basic GraphQL schema generation...")
        return _generate_basic_graphql_schema(swagger_json, output_path)

def _generate_basic_graphql_schema(swagger_json: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate a basic GraphQL schema from a Swagger/OpenAPI specification."""
    schema_lines = []
    
    # Add schema definition
    schema_lines.append("""# Generated GraphQL schema from Swagger/OpenAPI spec type Query {""")
    
    # Process paths to generate Query fields
    if "paths" in swagger_json:
        for path, methods in swagger_json["paths"].items():
            for method, details in methods.items():
                if method.lower() == "get":
                    # Convert path to camelCase field name
                    field_name = path.replace("/", "_").strip("_")
                    if not field_name:
                        field_name = "root"
                    
                    # Get response type
                    response_type = "JSON"
                    if "responses" in details and "200" in details["responses"]:
                        response = details["responses"]["200"]
                        if "schema" in response:
                            schema = response["schema"]
                            if "$ref" in schema:
                                ref = schema["$ref"]
                                type_name = ref.split("/")[-1]
                                response_type = type_name
                            elif "type" in schema and schema["type"] == "array" and "items" in schema:
                                if "$ref" in schema["items"]:
                                    ref = schema["items"]["$ref"]
                                    type_name = ref.split("/")[-1]
                                    response_type = f"[{type_name}]"
                    
                    # Add field to Query type
                    schema_lines.append(f"  {field_name}: {response_type}")
    
    # Close Query type
    schema_lines.append("}")
    
    # Process definitions to generate types
    definitions = swagger_json.get("definitions", {}) or swagger_json.get("components", {}).get("schemas", {})
    for type_name, type_def in definitions.items():
        schema_lines.append(f"\ntype {type_name} {{")
        
        if "properties" in type_def:
            for prop_name, prop_def in type_def["properties"].items():
                prop_type = "String"
                
                if "type" in prop_def:
                    if prop_def["type"] == "string":
                        prop_type = "String"
                    elif prop_def["type"] == "integer":
                        prop_type = "Int"
                    elif prop_def["type"] == "number":
                        prop_type = "Float"
                    elif prop_def["type"] == "boolean":
                        prop_type = "Boolean"
                    elif prop_def["type"] == "array":
                        if "items" in prop_def:
                            if "$ref" in prop_def["items"]:
                                ref = prop_def["items"]["$ref"]
                                item_type = ref.split("/")[-1]
                                prop_type = f"[{item_type}]"
                            elif "type" in prop_def["items"]:
                                item_type = prop_def["items"]["type"]
                                if item_type == "string":
                                    prop_type = "[String]"
                                elif item_type == "integer":
                                    prop_type = "[Int]"
                                elif item_type == "number":
                                    prop_type = "[Float]"
                                elif item_type == "boolean":
                                    prop_type = "[Boolean]"
                
                schema_lines.append(f"  {prop_name}: {prop_type}")
        
        schema_lines.append("}")
    
    # Join all lines to create the schema
    graphql_schema = "\n".join(schema_lines)
    
    # If output_path is provided, save the schema there as well
    if output_path:
        with open(output_path, "w") as f:
            f.write(graphql_schema)
    
    return graphql_schema
