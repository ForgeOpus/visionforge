# VisionForge Chatbot Setup Guide

This guide will help you set up and use the AI-powered chatbot feature in VisionForge.

## Overview

The VisionForge chatbot is an intelligent assistant that helps you build neural network architectures. It has two modes:

1. **Q&A Mode**: Answer questions about your workflow, explain concepts, and provide guidance
2. **Modification Mode**: Actively suggest and apply changes to your workflow

## Features

- Real-time conversation with context awareness
- Full workflow state visibility (nodes, edges, configurations)
- Interactive modification suggestions
- One-click application of AI-suggested changes
- Persistent chat history during the session
- Markdown formatting support in responses

## Setup Instructions

### 1. Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### 2. Configure Backend Environment

1. Navigate to the backend directory:
   ```bash
   cd project
   ```

2. Create a `.env` file (or copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file and add your Gemini API key:
   ```env
   # Gemini AI Configuration
   GEMINI_API_KEY=your-actual-api-key-here
   ```

   Replace `your-actual-api-key-here` with your actual API key.

### 3. Install Dependencies

Install the required Python package:

```bash
pip install -r requirements.txt
```

Or install the Gemini package directly:

```bash
pip install google-generativeai
```

### 4. Start the Backend Server

```bash
python manage.py runserver
```

The server should start on `http://localhost:8000`

### 5. Start the Frontend

In a separate terminal:

```bash
cd project/frontend
npm run dev
```

The frontend should start on `http://localhost:5173`

## Using the Chatbot

### Opening the Chat

Click the floating chat button in the bottom-right corner of the screen to open the chatbot panel.

### Chat Modes

#### Q&A Mode (Default)

When the **Modification Mode** toggle is OFF:
- Ask questions about your workflow
- Get explanations of neural network concepts
- Receive guidance on best practices
- Learn about available node types and configurations

**Example questions:**
- "What does my current architecture do?"
- "How do I add a convolutional layer?"
- "What is the BatchNorm2D layer used for?"
- "Can you explain the current connections in my workflow?"

#### Modification Mode

When the **Modification Mode** toggle is ON:
- AI can suggest specific changes to your workflow
- Receive actionable modification recommendations
- Apply changes with a single click

**Example requests:**
- "Add a Conv2D layer with 64 filters"
- "Add BatchNorm2D after the convolutional layer"
- "Remove the dropout layer"
- "Suggest improvements to reduce overfitting"

### Applying Modifications

When the AI suggests modifications:

1. The chat will display suggested changes with explanations
2. Each suggestion includes an **"Apply Change"** button
3. Click the button to automatically apply the modification to your workflow
4. You'll see a success notification confirming the change
5. The workflow canvas will update in real-time

### Workflow Context

The chatbot has full visibility into your current workflow:

- All nodes (layers) and their configurations
- All connections (edges) between nodes
- Node positions and arrangement
- Current architecture state

This context allows the AI to:
- Provide specific recommendations based on your architecture
- Suggest compatible layers
- Identify potential issues
- Reference specific nodes by name

## API Endpoints

The chatbot uses the following backend endpoints:

### POST /api/chat

Send a chat message with workflow context.

**Request:**
```json
{
  "message": "Add a Conv2D layer",
  "history": [
    {
      "role": "user",
      "content": "Previous message"
    },
    {
      "role": "assistant",
      "content": "Previous response"
    }
  ],
  "modificationMode": true,
  "workflowState": {
    "nodes": [...],
    "edges": [...]
  }
}
```

**Response:**
```json
{
  "response": "I'll help you add a Conv2D layer...",
  "modifications": [
    {
      "action": "add_node",
      "details": {
        "nodeType": "Conv2D",
        "config": {
          "in_channels": 3,
          "out_channels": 64,
          "kernel_size": 3
        },
        "position": {"x": 100, "y": 100}
      },
      "explanation": "Adding a Conv2D layer for feature extraction"
    }
  ]
}
```

### POST /api/suggestions

Get architecture improvement suggestions.

**Request:**
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
    "Consider adding BatchNorm2D after convolutional layers",
    "Add dropout layers to prevent overfitting",
    "Use ReLU activation for faster convergence"
  ]
}
```

## Modification Actions

The chatbot can suggest the following types of modifications:

### 1. Add Node
Adds a new layer to the workflow.

```json
{
  "action": "add_node",
  "details": {
    "nodeType": "Conv2D",
    "config": {
      "in_channels": 3,
      "out_channels": 64,
      "kernel_size": 3,
      "stride": 1,
      "padding": 1
    },
    "position": {"x": 100, "y": 200}
  },
  "explanation": "Adding convolutional layer for feature extraction"
}
```

### 2. Remove Node
Removes a layer from the workflow.

```json
{
  "action": "remove_node",
  "details": {
    "nodeId": "node-123"
  },
  "explanation": "Removing unnecessary layer"
}
```

### 3. Modify Node
Updates a layer's configuration.

```json
{
  "action": "modify_node",
  "details": {
    "nodeId": "node-123",
    "config": {
      "out_channels": 128,
      "kernel_size": 5
    }
  },
  "explanation": "Increasing filter size for better feature extraction"
}
```

### 4. Add Connection
Adds an edge between two nodes.

```json
{
  "action": "add_connection",
  "details": {
    "source": "node-123",
    "target": "node-456",
    "sourceHandle": "output",
    "targetHandle": "input"
  },
  "explanation": "Connecting layers to create data flow"
}
```

### 5. Remove Connection
Removes an edge between nodes.

```json
{
  "action": "remove_connection",
  "details": {
    "edgeId": "edge-123"
  },
  "explanation": "Removing invalid connection"
}
```

## Best Practices

### 1. Start with Q&A Mode
- Learn about the available features
- Understand your current workflow
- Get familiar with the chatbot's capabilities

### 2. Use Clear, Specific Requests
Instead of: "Make it better"
Try: "Add a Conv2D layer with 64 filters after the input"

### 3. Review Modifications Before Applying
- Read the explanation provided
- Understand what the change does
- Ensure it aligns with your goals

### 4. Iterative Refinement
- Apply one change at a time
- Review the results
- Continue the conversation to refine further

### 5. Provide Context
When asking questions, reference specific parts of your workflow:
- "What does the second Conv2D layer do?"
- "Should I add normalization after my pooling layer?"

## Troubleshooting

### Chatbot Not Responding

**Error:** "Gemini API key is not configured"

**Solution:**
1. Ensure you've set `GEMINI_API_KEY` in the `.env` file
2. Restart the Django server
3. Verify the API key is correct

### Connection Errors

**Error:** "I'm having trouble connecting to the server"

**Solution:**
1. Check that the backend server is running (`python manage.py runserver`)
2. Verify the frontend is configured with the correct API URL (`VITE_API_URL`)
3. Check for CORS errors in the browser console

### Modifications Not Applying

**Solution:**
1. Ensure you're in Modification Mode (toggle should be ON)
2. Check that the node IDs in suggestions match your workflow
3. Review browser console for errors

### API Rate Limits

If you exceed Gemini's rate limits:
1. Wait a few minutes before trying again
2. Consider upgrading your API plan
3. Reduce the frequency of requests

## Advanced Usage

### Custom System Prompts

The chatbot builds context-aware system prompts that include:
- Available node types
- Current workflow state
- Mode-specific instructions

This ensures the AI understands the VisionForge environment.

### Chat History Management

- Chat history is maintained in-memory during the session
- All previous messages provide context for new responses
- History is sent with each request for continuity
- Refreshing the page clears the history

### Workflow State Serialization

The workflow state is automatically serialized and sent with each message:
- Nodes: ID, type, label, configuration, position
- Edges: Source, target, handles
- This allows the AI to provide context-aware suggestions

## Security Considerations

1. **API Key Security**
   - Never commit `.env` files to version control
   - Keep your API key private
   - Rotate keys periodically

2. **Data Privacy**
   - Workflow data is sent to Google's Gemini API
   - Review Google's privacy policy for AI services
   - Don't include sensitive information in prompts

3. **Rate Limiting**
   - Implement request throttling for production use
   - Monitor API usage to avoid unexpected costs

## Future Enhancements

Planned features for the chatbot:

- [ ] Chat session persistence (save/load conversations)
- [ ] Export chat history
- [ ] Multi-user collaboration support
- [ ] Custom AI model selection (GPT-4, Claude, etc.)
- [ ] Voice input/output
- [ ] Batch modification application
- [ ] Undo/redo for AI changes
- [ ] Architecture templates from descriptions

## Support

For issues or questions:
1. Check this documentation
2. Review the console logs (browser and server)
3. Ensure all dependencies are installed
4. Verify API key configuration

## Example Use Cases

### 1. Building from Scratch

**User:** "I want to build an image classification model for CIFAR-10"

**AI Response:** "I'll help you create an architecture for CIFAR-10 classification..."

*[AI suggests adding Input, Conv2D, ReLU, MaxPool2D, etc.]*

### 2. Debugging Architecture

**User:** "Why isn't my model training well?"

**AI Response:** "Let me analyze your architecture... I notice you don't have any normalization layers. This can lead to training instability..."

### 3. Optimization

**User:** "How can I reduce overfitting?"

**AI Response:** "I can suggest several improvements:
1. Add Dropout layers (0.5 dropout rate)
2. Add BatchNorm2D after conv layers
3. Use data augmentation (in your training code)"

### 4. Learning

**User:** "What does the Flatten layer do?"

**AI Response:** "The Flatten layer reshapes multi-dimensional tensors into a 1D vector. For example, if you have a tensor of shape (batch, 64, 7, 7) from convolutional layers, Flatten will convert it to (batch, 3136), which can then be fed into Linear layers for classification..."

## Conclusion

The VisionForge chatbot is a powerful tool for building, understanding, and optimizing neural network architectures. By combining conversational AI with direct workflow manipulation, it provides an intuitive interface for both beginners and experts.

Remember to:
- Toggle Modification Mode when you want the AI to make changes
- Review suggestions before applying them
- Use the chat history to build context
- Experiment and iterate on your designs

Happy building!
