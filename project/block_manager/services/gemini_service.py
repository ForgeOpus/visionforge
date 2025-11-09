"""
Gemini AI Service for chat functionality and workflow modifications.
"""
import google.generativeai as genai
import json
import os
from typing import List, Dict, Any, Optional
from django.conf import settings


class GeminiChatService:
    """Service to handle Gemini AI chat interactions with workflow context."""

    def __init__(self):
        """Initialize Gemini with API key from environment."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

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
            data = node.get('data', {})
            label = data.get('label', 'Unlabeled')
            node_type_name = data.get('nodeType', data.get('blockType', 'unknown'))
            config = data.get('config', {})

            context_parts.append(f"  - {label} (ID: '{node_id}', NodeType: '{node_type_name}')")
            if config:
                config_str = ', '.join([f"{k}={v}" for k, v in config.items() if k != 'nodeType'])
                if config_str:
                    context_parts.append(f"    Config: {config_str}")

        if edges:
            context_parts.append("")
            context_parts.append("Connections:")
            for edge in edges:
                source = edge.get('source', '?')
                target = edge.get('target', '?')
                source_label = next((n.get('data', {}).get('label', source)
                                   for n in nodes if n.get('id') == source), source)
                target_label = next((n.get('data', {}).get('label', target)
                                   for n in nodes if n.get('id') == target), target)
                context_parts.append(f"  - {source_label} (ID: '{source}') → {target_label} (ID: '{target}')")

        return "\n".join(context_parts)

    def _build_system_prompt(self, modification_mode: bool, workflow_state: Optional[Dict[str, Any]]) -> str:
        """Build system prompt based on mode and workflow context."""
        base_prompt = """You are an AI assistant for VisionForge, a visual neural network architecture builder.

VisionForge allows users to create deep learning models by connecting nodes (blocks) in a visual workflow.

Available node types include:
- Input nodes: Input, DataLoader
- Convolutional layers: Conv1D, Conv2D, Conv3D
- Linear layers: Linear, Embedding
- Activation functions: ReLU, Softmax, Sigmoid, Tanh, LeakyReLU, Dropout
- Pooling layers: MaxPool2D, AvgPool2D, AdaptiveAvgPool2D
- Normalization: BatchNorm2D
- Merge operations: Concat, Add
- Utility: Flatten, Attention, Output, Loss

Each node has configurable parameters (e.g., channels, kernel_size, stride, etc.).
"""

        if modification_mode:
            mode_prompt = """
MODIFICATION MODE ENABLED:
You MUST provide actionable workflow modifications when users ask you to make changes.

CRITICAL: When a user asks you to add, remove, or modify nodes, you MUST:
1. Provide a natural language response explaining what you're doing
2. Include JSON modification blocks in your response using the exact format below

When the user says things like:
- "Add 2 input nodes" → Provide 2 separate add_node JSON blocks
- "Add a Conv2D layer" → Provide 1 add_node JSON block
- "Remove the dropout layer" → Provide 1 remove_node JSON block
- "Change kernel size to 5" → Provide 1 modify_node JSON block

MANDATORY FORMAT for each modification (include the ```json code fences):

```json
{
  "action": "add_node",
  "details": {
    "nodeType": "Input",
    "config": {"shape": "[1, 3, 224, 224]"},
    "position": {"x": 100, "y": 100}
  },
  "explanation": "Adding an Input node for image data"
}
```

FOR CONNECTIONS (add_connection):
To connect nodes, you MUST use the exact node IDs from the workflow context shown above.
Example:
```json
{
  "action": "add_connection",
  "details": {
    "source": "node-1234567890",
    "target": "node-9876543210",
    "sourceHandle": null,
    "targetHandle": null
  },
  "explanation": "Connecting the Input node to Conv2D layer"
}
```

IMPORTANT:
- ALWAYS wrap each modification in ```json ``` code fences
- Use exact node type names: Input, DataLoader, Conv2D, Linear, ReLU, etc.
- Use exact node IDs from the workflow context when creating connections
- For add_connection, source and target must be valid node IDs like 'node-1234567890'
- Provide reasonable default configurations
- If adding multiple nodes, include multiple separate JSON blocks
- The user will see "Apply Change" buttons for each modification
"""
        else:
            mode_prompt = """
Q&A MODE:
You are in question-answering mode. Help users understand their workflow, explain concepts, and provide guidance.
You cannot modify the workflow in this mode. If users want to make changes, suggest they enable modification mode.
"""

        workflow_context = self._format_workflow_context(workflow_state)

        return f"{base_prompt}\n{mode_prompt}\n{workflow_context}"

    def _format_chat_history(self, history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert chat history to Gemini format."""
        formatted_history = []

        for message in history:
            role = message.get('role', 'user')
            content = message.get('content', '')

            # Gemini uses 'user' and 'model' roles
            gemini_role = 'model' if role == 'assistant' else 'user'

            formatted_history.append({
                'role': gemini_role,
                'parts': [content]
            })

        return formatted_history

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        modification_mode: bool = False,
        workflow_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat message and get a response from Gemini.

        Args:
            message: User's message
            history: Previous chat messages [{'role': 'user'|'assistant', 'content': '...'}]
            modification_mode: Whether workflow modification is enabled
            workflow_state: Current workflow state (nodes and edges)

        Returns:
            {
                'response': str,
                'modifications': Optional[List[Dict]] - suggested workflow changes if any
            }
        """
        try:
            # Build system context
            system_prompt = self._build_system_prompt(modification_mode, workflow_state)

            # Format history for Gemini
            formatted_history = self._format_chat_history(history)

            # Always include system prompt with current workflow context
            # This ensures the AI always knows the current state and formatting requirements
            full_message = f"{system_prompt}\n\nUser: {message}"

            # Create chat session with history
            chat = self.model.start_chat(history=formatted_history)

            # Send message and get response
            response = chat.send_message(full_message)
            response_text = response.text

            # Try to extract JSON modifications from response
            modifications = self._extract_modifications(response_text)

            return {
                'response': response_text,
                'modifications': modifications if modification_mode else None
            }

        except Exception as e:
            return {
                'response': f"Error communicating with Gemini AI: {str(e)}",
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

            response = self.model.generate_content(prompt)
            response_text = response.text

            # Parse suggestions from numbered list
            import re
            suggestions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)', response_text, re.DOTALL)
            suggestions = [s.strip() for s in suggestions if s.strip()]

            return suggestions[:5]  # Return max 5 suggestions

        except Exception as e:
            return [f"Error generating suggestions: {str(e)}"]
