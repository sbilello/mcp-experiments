#!/usr/bin/env python3
"""
Swagger-to-GraphQL with Apollo Connectors CLI

This script provides a command-line interface for converting Swagger/OpenAPI specifications
to GraphQL schemas with Apollo Connector directives.

Usage:
  python swagger_to_graphql_cli.py --input <swagger_file> [--output <output_file>] [--verbose]
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Import the agent
from swagger_to_graphql_agent import run_swagger_to_graphql_agent

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

def main():
    """Main entry point for the CLI."""
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
    
    # Run the agent
    print("Converting Swagger/OpenAPI specification to GraphQL schema with Apollo Connector directives...")
    result = run_swagger_to_graphql_agent(swagger_spec)
    
    # Check for errors
    if result["error"]:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    # Save the schema
    save_graphql_schema(result["schema"], output_path)
    
    # Print verbose output if requested
    if args.verbose:
        print("\nAgent messages:")
        for i, message in enumerate(result["messages"]):
            print(f"Message {i+1}:")
            print(message[:200] + "..." if len(message) > 200 else message)
            print()
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()
