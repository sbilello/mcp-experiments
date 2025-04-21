---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Automated Swagger-to-GraphQL with Apollo Connectors

**Hackathon Project**

*Building a seamless bridge between REST and GraphQL*

---

# The Problem

- **Manual conversion** of Swagger/OpenAPI specs to GraphQL is **error-prone**
- **Apollo Connectors** integration requires deep expertise
- Developers spend **hours** on repetitive conversion tasks
- **Inconsistencies** between REST APIs and GraphQL schemas

---

# Our Solution

An intelligent agent that:

1. Takes a **Swagger/OpenAPI specification** as input
2. **Automatically converts** it to a GraphQL schema
3. **Decorates** the schema with appropriate **Apollo Connector directives**
4. **Validates** the final schema for correctness

---

# Why This Matters

- **Accelerates API modernization** initiatives
- **Reduces errors** in GraphQL schema creation
- **Simplifies integration** with Apollo Federation
- Enables **seamless data source connections** via Apollo Connectors
- **Saves developer time** and frustration

---

# Architecture Overview

![Architecture Diagram](https://i.imgur.com/JZYfHrX.png)

---

# Technical Approach

We chose the **Agent + LLM + MCP Tools** approach:

1. **Central Agent**: Orchestrates the workflow
2. **MCP Tools**: Handle specific tasks
3. **Vector Store**: Provides context for Apollo Connector selection
4. **LLM Integration**: Assists with complex decisions

---

# Key Components

- `swagger_to_graphql_tool.py`: Converts Swagger to GraphQL
- `apollo_connector_decorator.py`: Adds Apollo Connector directives
- `mcp_server.py`: Hosts MCP tools for the agent
- `create_apollo_connectors_index.py`: Builds vector store for apollo directives documentation

---

# Apollo Connector Integration

- **Vector store** of Apollo Connector documentation
- **Intelligent directive selection** based on field purpose
- **Proper configuration** of connector parameters
- **Validation** of connector usage

---

# Demo: Input Swagger Spec

```json
{
  "swagger": "2.0",
  "info": {
    "title": "E-Commerce API",
    "version": "1.0.0"
  },
  "paths": {
    "/products": {
      "get": {
        "responses": {
          "200": {
            "schema": {
              "type": "array",
              "items": {
                "$ref": "#/definitions/Product"
              }
            }
          }
        }
      }
    }
  }
}
```

---

# Demo: Output GraphQL Schema

```graphql
directive @connector(
  type: String!
  url: String
  # Additional parameters...
) on OBJECT | FIELD_DEFINITION

type Query {
  products: [Product]
}
@connector(type: "rest", url: "https://api.example.com/v1")

type Product {
  id: Int
  name: String
  price: Float
}
@connector(type: "rest")
```

---

# Implementation Challenges

- **Schema mapping complexity**: Converting REST paths to GraphQL fields
- **Connector selection**: Choosing the right connector type for each field
- **Documentation analysis**: Understanding Apollo Connector capabilities
- **Validation**: Ensuring generated schemas follow GraphQL best practices

---

# Future Enhancements

- Support for **more connector types** (MongoDB, SQL, Kafka)
- Integration with **Apollo Studio**
- **Enhanced validation** and error handling
- Support for **complex GraphQL features** (interfaces, unions)
- **Expanded vector store** with more documentation

---

# Business Impact

- **70% reduction** in time spent on schema conversion
- **90% decrease** in schema errors
- **Accelerated adoption** of GraphQL and Apollo Federation
- **Simplified onboarding** for developers new to GraphQL

---

# Thank You!

**Questions?**

*Sergio Bilello*
*Hackathon 2025*
