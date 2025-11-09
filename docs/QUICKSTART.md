# VisionForge Chatbot - Quick Start Guide

Get the AI chatbot up and running in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- Node.js 16+ installed
- Google account (for Gemini API key)

## Step 1: Get Your API Key (2 minutes)

1. Visit https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key to your clipboard

## Step 2: Configure Backend (1 minute)

1. Navigate to the project backend directory:
   ```bash
   cd project
   ```

2. Create a `.env` file:
   ```bash
   # On Windows
   copy .env.example .env

   # On macOS/Linux
   cp .env.example .env
   ```

3. Edit `.env` and paste your API key:
   ```env
   GEMINI_API_KEY=paste-your-key-here
   ```

## Step 3: Install Dependencies (1 minute)

```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Step 4: Start the Servers (1 minute)

**Terminal 1 - Backend:**
```bash
cd project
python manage.py runserver
```

**Terminal 2 - Frontend:**
```bash
cd project/frontend
npm run dev
```

## Step 5: Use the Chatbot!

1. Open your browser to `http://localhost:5173`
2. Click the **chat bubble icon** in the bottom-right corner
3. Start chatting!

### Quick Examples

**Q&A Mode (default):**
```
You: "How do I add a convolutional layer?"
AI: "You can add a Conv2D layer from the block palette..."
```

**Modification Mode (toggle ON):**
```
You: "Add a Conv2D layer with 64 filters"
AI: [Suggests modification with "Apply Change" button]
```

## That's It!

You now have a fully functional AI-powered chatbot that can:
- Answer questions about your neural network
- Suggest improvements
- Modify your workflow with one click

For detailed documentation, see [CHATBOT_SETUP.md](./CHATBOT_SETUP.md)

## Troubleshooting

### "API key is not configured"
- Check that `.env` file exists in the `project` folder
- Verify `GEMINI_API_KEY=your-key` is set correctly
- Restart the backend server

### "Connection error"
- Ensure both servers are running
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

### Chat not responding
- Check browser console (F12) for errors
- Verify your API key is valid
- Check internet connection

## Next Steps

1. Toggle **Modification Mode** to let AI modify your workflow
2. Ask the AI to suggest improvements to your architecture
3. Apply modifications with one click
4. Iterate and refine your neural network design

Happy building with VisionForge!
