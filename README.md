# Automated Swagger-to-GraphQL with Apollo Connectors

A hackathon prototype for automatically converting Swagger/OpenAPI specifications to GraphQL schemas decorated with Apollo Connector directives.

## Overview

This project automates the manual, error-prone process of converting Swagger/OpenAPI specifications to GraphQL schemas and integrating data sources via Apollo Connectors. It leverages the Agent Development Kit for orchestration and MCP tools for specific tasks.

### Key Features

- Parse and validate Swagger/OpenAPI specifications
- Generate GraphQL schemas from Swagger/OpenAPI specs
- Decorate GraphQL schemas with Apollo Connector directives
- Validate the generated GraphQL schemas
- Human-in-the-loop capability for error resolution

## Architecture

The solution follows the **Agent + LLM + MCP Tools** approach:

1. **Central Agent**: Orchestrates the entire workflow using LangGraph
2. **MCP Tools**: Handle specific tasks like parsing, conversion, and validation
3. **Vector Store**: Provides context for Apollo Connector directive selection
4. **LLM Integration**: Assists with complex decisions about connector selection

## Components

- `swagger_to_graphql_tool.py`: MCP tool for converting Swagger to GraphQL using @o/swagger-to-graphql
- `apollo_connector_decorator.py`: Decorates GraphQL schemas with Apollo Connector directives
- `swagger_to_graphql_agent.py`: LangGraph agent that orchestrates the workflow
- `swagger_to_graphql_cli.py`: Command-line interface for the prototype
- `mcp_server.py`: MCP server with tools for Swagger-to-GraphQL conversion and Apollo Connector decoration

## Prerequisites

- Python 3.10+
- Node.js and npm (for @o/swagger-to-graphql)
- Google API Key (for LLM and embeddings)

## Installation

1. Install the required Python packages:

```bash
pip install langgraph langchain langchain-google-genai langchain-community google-generativeai
```

2. Install the required npm package:

```bash
npm install -g @o/swagger-to-graphql
```

3. Set your Google API Key:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage

### Using the Direct Converter

```bash
python direct_swagger_to_graphql.py --input /path/to/swagger.json --output schema.graphql --verbose
```

### Example with Sample Swagger

```bash
python direct_swagger_to_graphql.py --input sample_swagger.json --output decorated_schema.graphql --verbose
```

### Using the Converter in Your Code

```python
from direct_swagger_to_graphql import generate_graphql_schema, decorate_with_apollo_connectors

# Read the Swagger spec
with open("swagger.json", "r") as f:
    swagger_json = json.load(f)

# Generate GraphQL schema
graphql_schema = generate_graphql_schema(swagger_json)

# Decorate with Apollo Connectors
decorated_schema = decorate_with_apollo_connectors(graphql_schema, swagger_json)

# Print the results
print(result["schema"])
```

## Example

Input: Swagger/OpenAPI specification
```json
{
  "swagger": "2.0",
  "info": {
    "title": "User API",
    "version": "1.0.0"
  },
  "paths": {
    "/users": {
      "get": {
        "summary": "Get all users",
        "responses": {
          "200": {
            "description": "A list of users",
            "schema": {
              "type": "array",
              "items": {
                "$ref": "#/definitions/User"
              }
            }
          }
        }
      }
    }
  },
  "definitions": {
    "User": {
      "type": "object",
      "properties": {
        "id": {
          "type": "integer"
        },
        "name": {
          "type": "string"
        }
      }
    }
  }
}
```

Output: GraphQL schema with Apollo Connector directives
```graphql
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

type Query {
  getUsers: [User]
}
@connector(type: "rest", url: "/api")

type User {
  id: Int
  name: String
}
@connector(type: "rest")
```

## Future Enhancements

- Support for more Apollo Connector types and configurations
- Integration with Apollo Studio
- Improved validation and error handling
- Support for complex GraphQL schema features (interfaces, unions, etc.)
- Expanded vector store with more Apollo Connector documentation
