#!/usr/bin/env python3
"""
Direct Swagger-to-GraphQL with Apollo Connectors

A standalone script that converts Swagger/OpenAPI specifications to GraphQL schemas
with Apollo Connector directives without external dependencies.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Union

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Swagger/OpenAPI specifications to GraphQL schemas with Apollo Connector directives"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to the Swagger/OpenAPI specification file"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Path to save the generated GraphQL schema (default: <input_file_base>.graphql)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    return parser.parse_args()

def read_swagger_spec(file_path: str) -> Dict[str, Any]:
    """Read a Swagger/OpenAPI specification from a file."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
            
        # Parse JSON
        swagger_json = json.loads(content)
        return swagger_json
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def save_graphql_schema(schema: str, output_path: str) -> None:
    """Save a GraphQL schema to a file."""
    try:
        with open(output_path, "w") as f:
            f.write(schema)
        print(f"GraphQL schema saved to: {output_path}")
    except Exception as e:
        print(f"Error saving schema: {e}")
        sys.exit(1)

def generate_graphql_schema(swagger_json: Dict[str, Any], verbose: bool = False) -> str:
    """Generate a GraphQL schema from a Swagger/OpenAPI specification."""
    if verbose:
        print("Generating GraphQL schema from Swagger/OpenAPI specification...")
    
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
    
    if verbose:
        print("GraphQL schema generation successful!")
        print(f"Schema size: {len(graphql_schema)} characters")
    
    return graphql_schema

def decorate_with_apollo_connectors(graphql_schema: str, swagger_json: Dict[str, Any], verbose: bool = False) -> str:
    """Decorate a GraphQL schema with Apollo Connector directives."""
    if verbose:
        print("Decorating GraphQL schema with Apollo Connector directives...")
    
    # Add Apollo Connector directive definitions
    connector_directive = """
# Apollo Connector directives
directive @connector(
  type: String!
  name: String
  url: String
  introspection: Boolean
  schema: String
  transforms: [String]
) on OBJECT | FIELD_DEFINITION

directive @tag(
  name: String!
) on FIELD_DEFINITION

directive @provides(
  fields: String!
) on FIELD_DEFINITION
"""
    
    # Split the schema into lines for processing
    schema_lines = graphql_schema.split("\n")
    decorated_lines = []
    
    # Add directive definitions at the beginning
    decorated_lines.append(connector_directive)
    
    # Process the schema line by line
    in_type_def = False
    current_type = None
    
    for line in schema_lines:
        # Skip comment lines
        if line.strip().startswith("#"):
            decorated_lines.append(line)
            continue
        
        # Check if this line starts a type definition
        if line.strip().startswith("type "):
            in_type_def = True
            type_parts = line.strip().split(" ")
            if len(type_parts) > 1:
                current_type = type_parts[1].split("{")[0].strip()
            decorated_lines.append(line)
        
        # Check if this line ends a type definition
        elif line.strip() == "}" and in_type_def:
            in_type_def = False
            
            # Add connector directive based on type
            if current_type == "Query":
                # Get base URL from Swagger if available
                base_url = "/"
                if "servers" in swagger_json and len(swagger_json["servers"]) > 0:
                    base_url = swagger_json["servers"][0].get("url", "/")
                elif "host" in swagger_json:
                    scheme = swagger_json.get("schemes", ["https"])[0]
                    base_path = swagger_json.get("basePath", "/")
                    base_url = f"{scheme}://{swagger_json['host']}{base_path}"
                
                # Add REST connector for Query type
                decorated_lines.append(f"  @connector(type: \"rest\", url: \"{base_url}\")")
            
            # Add connector for other types based on their properties
            elif current_type:
                # Default to REST connector for types
                decorated_lines.append(f"  @connector(type: \"rest\")")
            
            decorated_lines.append(line)
            current_type = None
        
        # Add other lines as is
        else:
            decorated_lines.append(line)
    
    # Join all lines to create the decorated schema
    decorated_schema = "\n".join(decorated_lines)
    
    if verbose:
        print("Schema decoration successful!")
        print(f"Decorated schema size: {len(decorated_schema)} characters")
    
    return decorated_schema

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Read the Swagger spec
    print(f"Reading Swagger/OpenAPI specification from: {args.input}")
    swagger_json = read_swagger_spec(args.input)
    
    # Determine the output path
    if args.output:
        output_path = args.output
    else:
        # Use the input filename with .graphql extension
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{base_name}.graphql"
    
    # Generate GraphQL schema
    print("Generating GraphQL schema from Swagger/OpenAPI specification...")
    graphql_schema = generate_graphql_schema(swagger_json, args.verbose)
    
    # Decorate the GraphQL schema
    print("Decorating GraphQL schema with Apollo Connector directives...")
    decorated_schema = decorate_with_apollo_connectors(graphql_schema, swagger_json, args.verbose)
    
    # Save the schema
    save_graphql_schema(decorated_schema, output_path)
    
    print("Conversion and decoration completed successfully!")
    
    # Print a sample of the schema if verbose
    if args.verbose:
        print("\nSchema sample:")
        schema_sample = decorated_schema[:500] + "..." if len(decorated_schema) > 500 else decorated_schema
        print(schema_sample)

if __name__ == "__main__":
    main()
