import json
import re
from typing import Dict, Any, List, Optional, Union

class ApolloConnectorDecorator:
    """
    A class to decorate GraphQL schema with Apollo Connector directives.
    """
    
    def __init__(self, connector_docs: Optional[List[str]] = None):
        """
        Initialize the decorator with optional connector documentation.
        
        Args:
            connector_docs: List of documentation strings about Apollo Connectors
        """
        self.connector_docs = connector_docs or []
        self.connector_types = {
            "rest": "@connector(type: \"rest\")",
            "sql": "@connector(type: \"sql\")",
            "graphql": "@connector(type: \"graphql\")",
            "mongodb": "@connector(type: \"mongodb\")",
            "kafka": "@connector(type: \"kafka\")",
            "grpc": "@connector(type: \"grpc\")"
        }
    
    def decorate_schema(self, graphql_schema: str, swagger_spec: Union[str, Dict[str, Any]]) -> str:
        """
        Decorate a GraphQL schema with Apollo Connector directives based on Swagger spec.
        
        Args:
            graphql_schema: The GraphQL schema as a string
            swagger_spec: The original Swagger/OpenAPI spec (JSON string or dict)
            
        Returns:
            str: The decorated GraphQL schema
        """
        # Parse swagger_spec if it's a string
        if isinstance(swagger_spec, str):
            try:
                swagger_spec = json.loads(swagger_spec)
            except json.JSONDecodeError:
                raise ValueError("Invalid Swagger spec JSON")
        
        # Add the directive definitions to the schema
        decorated_schema = self._add_directive_definitions(graphql_schema)
        
        # Analyze the Swagger spec to determine appropriate connectors
        connector_mappings = self._analyze_swagger_for_connectors(swagger_spec)
        
        # Apply the connector directives to the schema
        decorated_schema = self._apply_connector_directives(decorated_schema, connector_mappings)
        
        return decorated_schema
    
    def _add_directive_definitions(self, schema: str) -> str:
        """Add Apollo Connector directive definitions to the schema."""
        # Define the connector directive
        connector_directive = """
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
        # Check if schema already has directive definitions
        if "directive @connector" in schema:
            return schema
        
        # Add directive definitions at the beginning of the schema
        return connector_directive + schema
    
    def _analyze_swagger_for_connectors(self, swagger: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze Swagger spec to determine appropriate connectors for types.
        
        Returns:
            Dict mapping GraphQL type names to connector directives
        """
        connector_mappings = {}
        
        # Default to REST connector for the main Query type
        connector_mappings["Query"] = self.connector_types["rest"]
        
        # Analyze paths to determine connector types for other types
        if "paths" in swagger:
            base_url = swagger.get("servers", [{}])[0].get("url", "")
            connector_mappings["Query"] = f'@connector(type: "rest", url: "{base_url}")'
        
        # For each definition/schema in the Swagger spec, determine connector type
        definitions = swagger.get("definitions", {}) or swagger.get("components", {}).get("schemas", {})
        for type_name, type_def in definitions.items():
            # By default, use REST connector for types derived from Swagger
            connector_mappings[type_name] = self.connector_types["rest"]
            
            # TODO: More sophisticated analysis could be done here based on
            # type properties, formats, or other Swagger metadata
        
        return connector_mappings
    
    def _apply_connector_directives(self, schema: str, connector_mappings: Dict[str, str]) -> str:
        """Apply connector directives to types in the schema."""
        # Process the schema line by line
        lines = schema.split('\n')
        decorated_lines = []
        
        # Pattern to match type definitions
        type_pattern = re.compile(r'^type\s+(\w+)')
        
        for line in lines:
            decorated_lines.append(line)
            
            # Check if this line defines a type
            match = type_pattern.match(line.strip())
            if match:
                type_name = match.group(1)
                
                # If we have a connector mapping for this type, add the directive
                if type_name in connector_mappings:
                    # Add the connector directive on the next line with proper indentation
                    indent = len(line) - len(line.lstrip())
                    directive_line = ' ' * indent + connector_mappings[type_name]
                    decorated_lines.append(directive_line)
        
        return '\n'.join(decorated_lines)


def decorate_with_apollo_connectors(
    graphql_schema: str, 
    swagger_spec: Union[str, Dict[str, Any]],
    connector_docs: Optional[List[str]] = None
) -> str:
    """
    Decorate a GraphQL schema with Apollo Connector directives.
    
    Args:
        graphql_schema: The GraphQL schema as a string
        swagger_spec: The original Swagger/OpenAPI spec (JSON string or dict)
        connector_docs: Optional list of documentation strings about Apollo Connectors
        
    Returns:
        str: The decorated GraphQL schema
    """
    decorator = ApolloConnectorDecorator(connector_docs)
    return decorator.decorate_schema(graphql_schema, swagger_spec)
