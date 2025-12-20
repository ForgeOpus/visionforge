"""
OpenRouter AI Service for unified access to multiple AI providers.
Supports Gemini, GPT, and Claude models through a single API.
"""
import json
import os
import base64
from typing import List, Dict, Any, Optional
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from openai import OpenAI


class OpenRouterService:
    """Service to handle AI chat interactions via OpenRouter with workflow context."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize OpenRouter service with API key and model.

        Args:
            api_key: Optional API key for BYOK mode. If None, reads from environment.
            model: OpenRouter model identifier (e.g., 'google/gemini-3-flash-preview')
        """
        if api_key:
            # BYOK mode - use provided key
            final_api_key = api_key
        else:
            # DEV mode - use environment variable
            final_api_key = os.getenv('OPENROUTER_API_KEY')
            if not final_api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=final_api_key
        )

        # Use provided model or default to free tier Gemini
        self.model = model if model else 'google/gemini-3-flash-preview'

    def _format_workflow_context(self, workflow_state: Optional[Dict[str, Any]]) -> str:
        """Format workflow state into a readable context for the AI."""
        if not workflow_state:
            return "No workflow is currently loaded."

        nodes = workflow_state.get('nodes', [])
        edges = workflow_state.get('edges', [])

        context_parts = [
            "=== Current Workflow State ===",
            f"Total nodes: {len(nodes)}",
            f"Total connections: {len(edges)}",
            "",
            "Nodes in the workflow:"
        ]

        for node in nodes:
            node_id = node.get('id', 'unknown')
            node_type = node.get('type', 'unknown')
            position = node.get('position', {})
            data = node.get('data', {})
            label = data.get('label', 'Unlabeled')
            node_type_name = data.get('nodeType', data.get('blockType', 'unknown'))
            config = data.get('config', {})

            # Format node info with position
            pos_str = f"Position: x={position.get('x', 0)}, y={position.get('y', 0)}"
            context_parts.append(f"  - {label} (ID: '{node_id}', NodeType: '{node_type_name}', {pos_str})")
            if config:
                config_str = ', '.join([f"{k}={v}" for k, v in config.items() if k != 'nodeType'])
                if config_str:
                    context_parts.append(f"    Config: {config_str}")

        if edges:
            context_parts.append("")
            context_parts.append("Connections:")
            for edge in edges:
                edge_id = edge.get('id', '?')
                source = edge.get('source', '?')
                target = edge.get('target', '?')
                source_label = next((n.get('data', {}).get('label', source)
                                   for n in nodes if n.get('id') == source), source)
                target_label = next((n.get('data', {}).get('label', target)
                                   for n in nodes if n.get('id') == target), target)
                context_parts.append(f"  - {source_label} → {target_label} (Edge ID: '{edge_id}', Source: '{source}', Target: '{target}')")

        return "\n".join(context_parts)

    def _build_system_prompt(self, modification_mode: bool, workflow_state: Optional[Dict[str, Any]]) -> str:
        """Build system prompt based on mode and workflow context."""
        base_prompt = """You are an AI assistant for VisionForge, a visual neural network architecture builder.

VisionForge allows users to create deep learning models by connecting nodes (blocks) in a visual workflow.

=== AVAILABLE NODE TYPES AND THEIR CONFIGURATION SCHEMAS ===

INPUT NODES:
- "input": {"shape": "[1, 3, 224, 224]", "label": "Input"}
  - shape: tensor dimensions as string (required)
  - label: custom label (optional)

- "dataloader": {"dataset_name": "string", "batch_size": 32, "shuffle": true}

CONVOLUTIONAL LAYERS:
- "conv2d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1}
  - out_channels: REQUIRED (number of output channels)
  - kernel_size, stride, padding, dilation: optional (defaults shown)

- "conv1d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}
- "conv3d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}

LINEAR LAYERS:
- "linear": {"out_features": 10}
  - out_features: REQUIRED (output dimension)

- "embedding": {"num_embeddings": 1000, "embedding_dim": 128}
  - Both fields REQUIRED

ACTIVATION FUNCTIONS (no config needed, use empty object {}):
- "relu", "softmax", "sigmoid", "tanh", "leakyrelu": {}

POOLING LAYERS:
- "maxpool": {"kernel_size": 2, "stride": 2, "padding": 0}
- "avgpool": {"kernel_size": 2, "stride": 2, "padding": 0}
- "adaptiveavgpool": {"output_size": "[1, 1]"}

NORMALIZATION:
- "batchnorm": {"num_features": 64}
  - num_features: REQUIRED (must match input channels)

- "dropout": {"p": 0.5}
  - p: dropout probability (default 0.5)

MERGE OPERATIONS (no config needed):
- "concat": {}
- "add": {}

UTILITY:
- "flatten": {}
- "attention": {"embed_dim": 512, "num_heads": 8}
- "output": {} (no config)
- "loss": {"loss_type": "CrossEntropyLoss"}

CRITICAL RULES:
1. ALWAYS provide REQUIRED fields (marked above)
2. Use exact nodeType names in LOWERCASE: "input", "conv2d", "linear", "output", etc.
3. For conv2d, NEVER use "in_channels" - it's inferred from connections
4. Use empty config {} for nodes that don't need configuration
5. Provide reasonable defaults for optional fields
"""

        if modification_mode:
            mode_prompt = """
MODIFICATION MODE ENABLED:
You MUST provide actionable workflow modifications when users ask you to make changes.

CRITICAL INSTRUCTION - BE PRECISE AND MINIMAL:
- ONLY add/modify/remove what the user EXPLICITLY requests
- DO NOT be creative or add extra nodes unless asked
- Follow the user's exact specifications to the letter
- Provide a brief natural language response
- Include ONLY the JSON blocks for what was requested

Examples of CORRECT responses:
- User: "Add 2 input nodes" → Provide EXACTLY 2 add_node blocks for input, NOTHING MORE
- User: "Add a Conv2D layer" → Provide EXACTLY 1 add_node block for conv2d, NOTHING MORE
- User: "input connects to conv2d connects to output" → Provide EXACTLY 3 add_node blocks (input, conv2d, output), mention connections will be added after nodes exist
- User: "Remove dropout" → Provide EXACTLY 1 remove_node block
- User: "Duplicate the ReLU" → Provide EXACTLY 1 duplicate_node block
- User: "Change kernel to 5" → Provide EXACTLY 1 modify_node block
- User: "Move conv2d down" → Provide EXACTLY 1 modify_node block with position
- User: "Rename input to 'Image Data'" → Provide EXACTLY 1 modify_node block with label

MANDATORY FORMAT for each modification (include the ```json code fences):

FOR ADDING NODES:
```json
{
  "action": "add_node",
  "details": {
    "nodeType": "input",
    "config": {"shape": "[1, 3, 224, 224]"},
    "position": {"x": 100, "y": 100}
  },
  "explanation": "Adding an Input node for image data"
}
```

FOR REMOVING NODES:
Use the exact node ID from the workflow context:
```json
{
  "action": "remove_node",
  "details": {
    "id": "conv-1234567890"
  },
  "explanation": "Removing the Conv2D layer"
}
```

FOR DUPLICATING NODES:
Creates a copy of an existing node with the same configuration:
```json
{
  "action": "duplicate_node",
  "details": {
    "id": "relu-1234567890"
  },
  "explanation": "Duplicating the ReLU activation"
}
```

FOR MODIFYING NODES:
Use modify_node to update node configuration, position, or label:
- To update config: include "id" and "config" fields
- To move a node: include "id" and "position" fields
- To rename a node: include "id" and "label" fields
- You can update multiple properties at once

Example (updating config):
```json
{
  "action": "modify_node",
  "details": {
    "id": "conv-1234567890",
    "config": {"kernel_size": 5, "padding": 2}
  },
  "explanation": "Changing kernel size to 5 and padding to 2"
}
```

FOR CONNECTIONS:
```json
{
  "action": "add_connection",
  "details": {
    "source": "node-1234567890",
    "target": "node-9876543210",
    "sourceHandle": null,
    "targetHandle": null
  },
  "explanation": "Connecting Input to Conv2D"
}
```

IMPORTANT RULES:
- ALWAYS wrap each modification in ```json ``` code fences
- Use exact node type names in LOWERCASE: input, dataloader, conv2d, linear, relu, etc.
- For node operations (remove, duplicate, modify), ALWAYS use node IDs from the current workflow context
- Provide only what user explicitly requests
"""
        else:
            mode_prompt = """
Q&A MODE:
You are in question-answering mode. Help users understand their workflow, explain concepts, and provide guidance.
You cannot modify the workflow in this mode. If users want to make changes, suggest they enable modification mode.
"""

        workflow_context = self._format_workflow_context(workflow_state)

        return f"{base_prompt}\n{mode_prompt}\n{workflow_context}"

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        modification_mode: bool = False,
        workflow_state: Optional[Dict[str, Any]] = None,
        uploaded_file: Optional[UploadedFile] = None
    ) -> Dict[str, Any]:
        """
        Send a chat message and get a response via OpenRouter.

        Args:
            message: User's message
            history: Previous chat messages [{'role': 'user'|'assistant', 'content': '...'}]
            modification_mode: Whether workflow modification is enabled
            workflow_state: Current workflow state (nodes and edges)
            uploaded_file: Optional file upload (will be base64 encoded)

        Returns:
            {
                'response': str,
                'modifications': Optional[List[Dict]] - suggested workflow changes if any
            }
        """
        try:
            # Build system context
            system_prompt = self._build_system_prompt(modification_mode, workflow_state)

            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add history
            for msg in history:
                messages.append({
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', '')
                })

            # Handle file upload (if present)
            if uploaded_file:
                # For vision models, encode image as base64
                file_content = uploaded_file.read()
                file_base64 = base64.b64encode(file_content).decode('utf-8')
                file_type = uploaded_file.content_type or 'image/jpeg'

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file_type};base64,{file_base64}"
                            }
                        }
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": message
                })

            # Call OpenRouter API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            response_text = response.choices[0].message.content

            # Try to extract JSON modifications from response
            modifications = self._extract_modifications(response_text)

            return {
                'response': response_text,
                'modifications': modifications if modification_mode else None
            }

        except Exception as e:
            return {
                'response': f"Error communicating with AI: {str(e)}",
                'modifications': None
            }

    def _extract_modifications(self, response_text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract JSON modification suggestions from AI response."""
        try:
            # Look for JSON code blocks
            import re
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if matches:
                modifications = []
                for match in matches:
                    try:
                        mod = json.loads(match)
                        if 'action' in mod:
                            modifications.append(mod)
                    except json.JSONDecodeError:
                        continue

                return modifications if modifications else None

            return None

        except Exception:
            return None

    def generate_suggestions(
        self,
        workflow_state: Dict[str, Any]
    ) -> List[str]:
        """
        Generate architecture improvement suggestions based on current workflow.

        Args:
            workflow_state: Current workflow state (nodes and edges)

        Returns:
            List of suggestion strings
        """
        try:
            workflow_context = self._format_workflow_context(workflow_state)

            prompt = f"""Analyze this neural network architecture and provide 3-5 specific improvement suggestions.

{workflow_context}

Provide suggestions as a numbered list. Focus on:
1. Architecture improvements (missing layers, better configurations)
2. Common best practices
3. Potential issues or bottlenecks
4. Training optimization opportunities

Format your response as a simple numbered list."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specializing in neural network architectures."},
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = response.choices[0].message.content

            # Parse suggestions from numbered list
            import re
            suggestions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)', response_text, re.DOTALL)
            suggestions = [s.strip() for s in suggestions if s.strip()]

            return suggestions[:5]  # Return max 5 suggestions

        except Exception as e:
            return [f"Error generating suggestions: {str(e)}"]
