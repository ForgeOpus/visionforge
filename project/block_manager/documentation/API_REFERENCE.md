# VisionForge Backend API Reference

## Overview

The VisionForge backend provides RESTful API endpoints for managing AI model building projects, validating architectures, and exporting code.

**Base URL:** `http://localhost:8000/api/`

**Authentication:** Not required (local development only)

---

## Table of Contents

1. [Project Management](#project-management)
2. [Architecture Management](#architecture-management)
3. [Validation](#validation)
4. [Code Export](#code-export)
5. [AI Assistance](#ai-assistance)
6. [Data Models](#data-models)
7. [Error Handling](#error-handling)

---

## Project Management

### List All Projects

**Endpoint:** `GET /api/projects/`

**Description:** Retrieve a list of all projects

**Response:**
```json
{
  "projects": [
    {
      "id": "uuid",
      "name": "My Vision Model",
      "description": "Custom CNN architecture",
      "framework": "pytorch",
      "created_at": "2025-11-08T10:30:00Z",
      "updated_at": "2025-11-08T14:45:00Z"
    }
  ]
}
```

---

### Create Project

**Endpoint:** `POST /api/projects/`

**Description:** Create a new project

**Request Body:**
```json
{
  "name": "My New Model",
  "description": "Transformer-based architecture",
  "framework": "pytorch"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "name": "My New Model",
  "description": "Transformer-based architecture",
  "framework": "pytorch",
  "created_at": "2025-11-08T15:00:00Z",
  "updated_at": "2025-11-08T15:00:00Z"
}
```

---

### Get Project Details

**Endpoint:** `GET /api/projects/{id}/`

**Description:** Retrieve detailed information about a specific project, including its architecture

**Response:**
```json
{
  "id": "uuid",
  "name": "My Vision Model",
  "description": "Custom CNN architecture",
  "framework": "pytorch",
  "architecture": {
    "id": "uuid",
    "canvas_state": {
      "nodes": [...],
      "edges": [...]
    },
    "is_valid": true,
    "validation_errors": [],
    "blocks": [...],
    "connections": [...],
    "created_at": "2025-11-08T10:30:00Z",
    "updated_at": "2025-11-08T14:45:00Z"
  },
  "created_at": "2025-11-08T10:30:00Z",
  "updated_at": "2025-11-08T14:45:00Z"
}
```

---

### Update Project

**Endpoint:** `PUT /api/projects/{id}/` or `PATCH /api/projects/{id}/`

**Description:** Update project metadata

**Request Body:**
```json
{
  "name": "Updated Model Name",
  "description": "Updated description"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "Updated Model Name",
  "description": "Updated description",
  "framework": "pytorch",
  "created_at": "2025-11-08T10:30:00Z",
  "updated_at": "2025-11-08T16:00:00Z"
}
```

---

### Delete Project

**Endpoint:** `DELETE /api/projects/{id}/`

**Description:** Delete a project and all associated data

**Response:** `204 No Content`

---

## Architecture Management

### Save Architecture

**Endpoint:** `POST /api/projects/{id}/save-architecture/`

**Description:** Save the current canvas state for a project

**Request Body:**
```json
{
  "nodes": [
    {
      "id": "node_1",
      "type": "input",
      "position": { "x": 100, "y": 100 },
      "data": {
        "blockType": "input",
        "label": "Input Layer",
        "category": "input",
        "config": {
          "inputShape": {"dims": [1, 3, 224, 224]}
        },
        "inputShape": {"dims": [1, 3, 224, 224]},
        "outputShape": {"dims": [1, 3, 224, 224]}
      }
    },
    {
      "id": "node_2",
      "type": "conv2d",
      "position": { "x": 300, "y": 100 },
      "data": {
        "blockType": "conv2d",
        "label": "Conv2D Layer",
        "category": "basic",
        "config": {
          "out_channels": 64,
          "kernel_size": 3,
          "stride": 1,
          "padding": 1
        }
      }
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "node_1",
      "target": "node_2",
      "sourceHandle": "",
      "targetHandle": ""
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "architecture_id": "uuid",
  "validation": {
    "is_valid": true,
    "errors": []
  }
}
```

---

### Load Architecture

**Endpoint:** `GET /api/projects/{id}/load-architecture/`

**Description:** Load the saved canvas state for a project

**Response:**
```json
{
  "nodes": [...],
  "edges": [...]
}
```

---

## Validation

### Validate Architecture

**Endpoint:** `POST /api/validate/`

**Description:** Validate model architecture for correctness and infer tensor shapes

**Request Body:**
```json
{
  "nodes": [
    {
      "id": "node_1",
      "data": {
        "blockType": "input",
        "config": {
          "inputShape": {"dims": [1, 3, 224, 224]}
        }
      }
    }
  ],
  "edges": []
}
```

**Response:**
```json
{
  "isValid": true,
  "errors": [],
  "warnings": [
    {
      "message": "Block is not connected to the graph",
      "nodeId": "node_1",
      "type": "warning",
      "suggestion": "Connect this block or remove it from the canvas"
    }
  ],
  "inferred_shapes": {
    "node_1": {
      "inputShape": {"dims": [1, 3, 224, 224]},
      "outputShape": {"dims": [1, 3, 224, 224]}
    }
  }
}
```

**Validation Checks:**
- At least one input block exists
- No circular dependencies
- All required parameters are configured
- Blocks with multiple inputs are merge blocks (concat/add)
- No orphaned blocks (except warnings)

**Error Response Example:**
```json
{
  "isValid": false,
  "errors": [
    {
      "message": "Linear layer requires out_features parameter",
      "nodeId": "node_3",
      "type": "error",
      "suggestion": "Configure the number of output features in the configuration panel"
    }
  ],
  "warnings": []
}
```

---

## Code Export

### Export Model Code

**Endpoint:** `POST /api/export/`

**Description:** Generate PyTorch or TensorFlow code from the architecture

**Request Body:**
```json
{
  "nodes": [...],
  "edges": [...],
  "format": "pytorch"
}
```

**Parameters:**
- `nodes` (required): Array of node objects
- `edges` (required): Array of edge objects
- `format` (required): One of `"pytorch"`, `"tensorflow"`, or `"onnx"`

**Response:**
```json
{
  "code": "import torch\nimport torch.nn as nn\n\nclass GeneratedModel(nn.Module):\n    ..."
}
```

**Note:** Current implementation returns stub code. Full code generation will be implemented in Phase 5.

---

## AI Assistance

### Send Chat Message

**Endpoint:** `POST /api/chat/`

**Description:** Send a message to the AI assistant for help

**Request Body:**
```json
{
  "message": "How do I add a dropout layer?",
  "history": []
}
```

**Response:**
```json
{
  "response": "I received your message: 'How do I add a dropout layer?'. AI chat integration coming soon!"
}
```

**Note:** Current implementation returns stub responses. Full AI integration will be added in future phases.

---

### Get Architecture Suggestions

**Endpoint:** `POST /api/suggestions/`

**Description:** Get suggestions for improving the current architecture

**Request Body:**
```json
{
  "nodes": [...],
  "edges": [...]
}
```

**Response:**
```json
{
  "suggestions": [
    "Consider adding a Dropout layer to prevent overfitting",
    "Add Batch Normalization after convolutional layers",
    "Use ReLU activation for faster convergence"
  ]
}
```

---

## Data Models

### Project
```python
{
    "id": "uuid",
    "name": "string",
    "description": "text",
    "framework": "choice['pytorch', 'tensorflow']",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

### ModelArchitecture
```python
{
    "id": "uuid",
    "project_id": "fk->Project",
    "canvas_state": "json",
    "is_valid": "boolean",
    "validation_errors": "json",
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

### Block
```python
{
    "id": "uuid",
    "architecture_id": "fk->ModelArchitecture",
    "node_id": "string",
    "block_type": "string",
    "position_x": "float",
    "position_y": "float",
    "config": "json",
    "input_shape": "json",
    "output_shape": "json",
    "created_at": "datetime"
}
```

### Connection
```python
{
    "id": "uuid",
    "architecture_id": "fk->ModelArchitecture",
    "edge_id": "string",
    "source_block": "fk->Block",
    "target_block": "fk->Block",
    "source_handle": "string",
    "target_handle": "string",
    "is_valid": "boolean",
    "created_at": "datetime"
}
```

---

## Error Handling

### HTTP Status Codes

- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `204 No Content` - Resource deleted successfully
- `400 Bad Request` - Invalid request data
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

### Error Response Format

```json
{
  "error": "Error message describing what went wrong",
  "details": {
    "field": ["Specific field error"]
  }
}
```

---

## Shape Inference Rules

The dimension inference engine automatically computes tensor shapes based on block types:

| Block Type | Input → Output |
|-----------|----------------|
| Input | User-defined → User-defined |
| Linear | [B, F_in] → [B, F_out] |
| Conv2D | [B, C_in, H, W] → [B, C_out, H', W'] |
| Flatten | [B, ...] → [B, product] |
| MaxPool | [B, C, H, W] → [B, C, H', W'] |
| Dropout | [B, ...] → [B, ...] (preserves) |
| BatchNorm | [B, ...] → [B, ...] (preserves) |
| ReLU/Activation | [B, ...] → [B, ...] (preserves) |
| Concat | Multiple inputs → Combined along dimension |
| Add | [B, ...] + [B, ...] → [B, ...] (element-wise) |

Where:
- `B` = Batch size
- `F` = Features
- `C` = Channels
- `H, W` = Height, Width
- `H', W'` = Computed based on kernel, stride, padding

---

## CORS Configuration

The backend is configured to accept requests from:
- `http://localhost:3000` (React dev server)
- `http://localhost:5173` (Vite dev server)

---

## Development Notes

1. **Database:** SQLite for local development
2. **Authentication:** Not implemented (single-user local application)
3. **Rate Limiting:** Not implemented
4. **Pagination:** Not implemented (assumes small number of projects)
5. **File Upload:** Not currently supported
6. **WebSockets:** Not currently supported

---

## Future Enhancements

- Full code generation for PyTorch and TensorFlow
- AI-powered chat assistance
- Advanced architecture suggestions
- Model performance profiling
- Export to additional formats (ONNX, CoreML)
- Training script generation
- Dataset integration

---

## Support

For issues or questions, please refer to the project documentation or create an issue in the repository.
