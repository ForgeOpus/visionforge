# AI Provider Implementation Summary

## Overview

VisionForge now supports **two AI providers** for the chatbot functionality:
1. **Gemini** (Google's Generative AI) - Original provider
2. **Claude** (Anthropic's Claude AI) - New addition

Users can switch between providers using a simple environment variable configuration.

---

## Files Created

### 1. Claude Service (`project/block_manager/services/claude_service.py`)
- **Purpose:** Implements Claude AI integration mirroring Gemini's functionality
- **Key Features:**
  - Chat with conversation history
  - Workflow modification suggestions
  - File upload support (images, PDFs, text)
  - Architecture generation from files
  - Improvement suggestions

**Model Used:** `claude-3-5-sonnet-20241022` (Latest Claude Sonnet)

**Key Differences from Gemini:**
- Uses Anthropic SDK (`anthropic` package)
- File handling: Base64 encoding for images/PDFs instead of File API
- Message format: Direct user/assistant roles (no conversion needed)
- Response format: `response.content[0].text` instead of `response.text`

### 2. AI Service Factory (`project/block_manager/services/ai_service_factory.py`)
- **Purpose:** Provider selection and instantiation
- **Methods:**
  - `create_service()`: Returns appropriate service based on `AI_PROVIDER` env var
  - `get_provider_name()`: Returns human-readable provider name

**Error Handling:**
- Validates `AI_PROVIDER` value (must be 'gemini' or 'claude')
- Propagates API key errors from individual services

---

## Files Modified

### 1. Requirements (`project/requirements.txt`)
**Added:**
```
anthropic>=0.39.0
```

### 2. Environment Configuration

#### `.env`
```env
# AI Provider Configuration
AI_PROVIDER=gemini  # or 'claude'

# Gemini AI Configuration
GEMINI_API_KEY=your-key-here

# Claude AI Configuration
ANTHROPIC_API_KEY=your-key-here
```

#### `.env.example`
Same structure with placeholder values.

### 3. Chat Views (`project/block_manager/views/chat_views.py`)

**Changes:**
- Replaced direct `GeminiChatService` import with `AIServiceFactory`
- Updated `chat_message()` endpoint:
  - Uses factory to create service
  - Handles file uploads differently per provider:
    - **Gemini:** Uploads to Gemini File API → passes `gemini_file` param
    - **Claude:** Reads file content locally → passes `file_content` param
  - Provider-agnostic error messages
- Updated `get_suggestions()` endpoint:
  - Uses factory instead of direct service instantiation
  - Generic error messages

### 4. Documentation (`docs/CHATBOT_SETUP.md`)

**Additions:**
- Provider selection guide
- Separate setup instructions for Gemini and Claude
- API key acquisition for both providers
- Provider comparison section
- Troubleshooting for provider switching
- Updated security and privacy sections

---

## Configuration Guide

### Using Gemini (Default)

1. **Get API Key:**
   - Visit: https://aistudio.google.com/app/apikey
   - Create free API key

2. **Configure `.env`:**
   ```env
   AI_PROVIDER=gemini
   GEMINI_API_KEY=AIzaSy...
   ```

3. **Restart server:**
   ```bash
   python manage.py runserver
   ```

### Using Claude

1. **Get API Key:**
   - Visit: https://console.anthropic.com/
   - Create account and generate API key

2. **Configure `.env`:**
   ```env
   AI_PROVIDER=claude
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **Install package:**
   ```bash
   pip install anthropic
   ```

4. **Restart server:**
   ```bash
   python manage.py runserver
   ```

### Switching Providers

Simply update `AI_PROVIDER` in `.env` and restart the server. No code changes needed!

---

## Technical Implementation Details

### Architecture Pattern: Factory + Strategy

```
User Request
    ↓
chat_views.py
    ↓
AIServiceFactory.create_service()
    ↓
    ├─→ GeminiChatService (if AI_PROVIDER=gemini)
    └─→ ClaudeChatService (if AI_PROVIDER=claude)
    ↓
AI Provider API
    ↓
Response → Frontend
```

### Common Interface

Both services implement the same interface:
```python
class ChatServiceInterface:
    def chat(message, history, modification_mode, workflow_state, **kwargs)
    def generate_suggestions(workflow_state)
    def _format_workflow_context(workflow_state)
    def _build_system_prompt(modification_mode, workflow_state)
    def _extract_modifications(response_text)
```

### File Upload Handling

**Gemini Approach:**
1. Save uploaded file to temp location
2. Upload to Gemini File API using `genai.upload_file()`
3. Pass file object to model
4. Clean up temp file

**Claude Approach:**
1. Read file content directly from Django's UploadedFile
2. Encode images/PDFs as base64
3. Include in message content array
4. No temp file needed

### Response Parsing

Both services use identical regex pattern to extract JSON modifications:
```python
json_pattern = r'```json\s*(\{.*?\})\s*```'
```

This ensures consistent modification format regardless of provider.

---

## API Compatibility

### Request Format (Same for Both)
```json
{
  "message": "Add a Conv2D layer",
  "history": [{"role": "user", "content": "..."}],
  "modificationMode": true,
  "workflowState": {"nodes": [...], "edges": [...]}
}
```

### Response Format (Same for Both)
```json
{
  "response": "AI response text...",
  "modifications": [
    {
      "action": "add_node",
      "details": {...},
      "explanation": "..."
    }
  ]
}
```

**Frontend compatibility:** No changes needed! The response format is identical.

---

## Error Handling

### Configuration Errors

**Invalid Provider:**
```
ValueError: Invalid AI_PROVIDER: 'gpt4'. Must be 'gemini' or 'claude'.
```

**Missing API Key (Gemini):**
```
ValueError: GEMINI_API_KEY environment variable is not set
```

**Missing API Key (Claude):**
```
ValueError: ANTHROPIC_API_KEY environment variable is not set
```

### Runtime Errors

Both services handle:
- API communication failures
- Rate limiting
- Invalid file uploads
- Malformed responses

Errors are logged and returned as user-friendly messages.

---

## Testing Checklist

- [x] Gemini provider works with chat
- [x] Gemini provider works with file uploads
- [x] Gemini provider works with suggestions
- [ ] Claude provider works with chat (requires API key)
- [ ] Claude provider works with file uploads (requires API key)
- [ ] Claude provider works with suggestions (requires API key)
- [x] Provider switching works
- [x] Error handling for missing API keys
- [x] Error handling for invalid provider
- [x] Documentation updated

---

## Provider Comparison

| Feature | Gemini | Claude |
|---------|--------|--------|
| **Model** | gemini-2.0-flash | claude-3-5-sonnet-20241022 |
| **Speed** | Very Fast | Fast |
| **Free Tier** | ✅ Yes | ❌ No |
| **Image Support** | ✅ Yes | ✅ Yes |
| **PDF Support** | ✅ Yes | ✅ Yes |
| **Max Tokens** | 8192 | 4096 (configurable) |
| **Reasoning** | Good | Excellent |
| **Code Understanding** | Good | Excellent |
| **Rate Limit (Free)** | 15 RPM | N/A |

---

## Future Enhancements

### Potential Additions:
1. **OpenAI GPT-4** support
2. **Provider-specific features:**
   - Gemini: Grounding with Google Search
   - Claude: Extended context (200k tokens)
3. **Provider fallback:** If one fails, try another
4. **Cost tracking:** Monitor API usage per provider
5. **A/B testing:** Compare response quality
6. **Provider-specific prompts:** Optimize for each model's strengths

### Extension Pattern:
```python
# Add new provider:
# 1. Create service class: NewProviderChatService
# 2. Update AIServiceFactory.create_service()
# 3. Add env vars: NEW_PROVIDER_API_KEY
# 4. Update documentation
```

---

## Security Considerations

1. **API Keys:**
   - Stored in `.env` (git-ignored)
   - Never exposed to frontend
   - Validated at service initialization

2. **Data Privacy:**
   - Workflow data sent to external APIs
   - User should review provider privacy policies
   - No sensitive data should be in workflows

3. **Rate Limiting:**
   - Implement request throttling in production
   - Monitor costs (especially for Claude)
   - Consider caching common responses

---

## Troubleshooting

### Issue: "AI service not properly configured"
**Cause:** Missing or invalid `AI_PROVIDER` or API key
**Solution:**
1. Check `.env` file has `AI_PROVIDER=gemini` or `AI_PROVIDER=claude`
2. Verify corresponding API key is set
3. Restart Django server

### Issue: Provider not switching
**Cause:** Server not restarted after `.env` change
**Solution:** Always restart Django after changing environment variables

### Issue: File uploads failing with Claude
**Cause:** Unsupported file type or size
**Solution:**
- Check file is image (PNG, JPG, WEBP, GIF) or PDF
- Ensure file is under 10MB
- Review error logs for details

---

## Summary

This implementation provides:
- ✅ **Flexibility:** Easy provider switching via config
- ✅ **Consistency:** Same API interface for both providers
- ✅ **Maintainability:** Factory pattern for easy extension
- ✅ **Reliability:** Comprehensive error handling
- ✅ **Documentation:** Complete setup and usage guides

**No frontend changes required** - the implementation is completely transparent to the client.

Users can now choose the AI provider that best fits their needs, budget, and preferences!
