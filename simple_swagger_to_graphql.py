#!/usr/bin/env python3
"""
Simple Swagger-to-GraphQL with Apollo Connectors

This script provides a simplified approach to convert Swagger/OpenAPI specifications
to GraphQL schemas with Apollo Connector directives without external dependencies.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Import our custom modules
from swagger_to_graphql_tool import swagger_to_graphql
from apollo_connector_decorator import decorate_with_apollo_connectors

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

def read_swagger_spec(file_path: str) -> str:
    """Read a Swagger/OpenAPI specification from a file."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
            
        # Validate that it's valid JSON
        json.loads(content)
        return content
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

def convert_swagger_to_graphql(swagger_spec: str, verbose: bool = False) -> str:
    """Convert a Swagger/OpenAPI specification to a GraphQL schema."""
    try:
        if verbose:
            print("Converting Swagger/OpenAPI specification to GraphQL schema...")
        
        # Convert Swagger to GraphQL
        graphql_schema = swagger_to_graphql(swagger_spec)
        
        if verbose:
            print("Conversion successful!")
            print(f"GraphQL schema size: {len(graphql_schema)} characters")
        
        return graphql_schema
    except Exception as e:
        print(f"Error converting Swagger to GraphQL: {e}")
        sys.exit(1)

def decorate_graphql_schema(graphql_schema: str, swagger_spec: str, verbose: bool = False) -> str:
    """Decorate a GraphQL schema with Apollo Connector directives."""
    try:
        if verbose:
            print("Decorating GraphQL schema with Apollo Connector directives...")
        
        # Decorate the schema
        decorated_schema = decorate_with_apollo_connectors(graphql_schema, swagger_spec)
        
        if verbose:
            print("Decoration successful!")
            print(f"Decorated schema size: {len(decorated_schema)} characters")
        
        return decorated_schema
    except Exception as e:
        print(f"Error decorating GraphQL schema: {e}")
        sys.exit(1)

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Read the Swagger spec
    print(f"Reading Swagger/OpenAPI specification from: {args.input}")
    swagger_spec = read_swagger_spec(args.input)
    
    # Determine the output path
    if args.output:
        output_path = args.output
    else:
        # Use the input filename with .graphql extension
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{base_name}.graphql"
    
    # Convert Swagger to GraphQL
    print("Converting Swagger/OpenAPI specification to GraphQL schema...")
    graphql_schema = convert_swagger_to_graphql(swagger_spec, args.verbose)
    
    # Decorate the GraphQL schema
    print("Decorating GraphQL schema with Apollo Connector directives...")
    decorated_schema = decorate_graphql_schema(graphql_schema, swagger_spec, args.verbose)
    
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
